import copy
import importlib
import json
import math
import os
import pathlib
import pickle
import sys
import time

import numpy as np

import models


import nevergrad
import numpy
import ray
import torch

import diagnose_model
import replay_buffer
import self_play
import shared_storage
import trainer
import wandb
import argparse
from utils import refresh, print_config
from self_play import SelfPlay

import networks.muzero_network as mz_net
from muzero_logger import logging_loop

class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        # >>> muzero = MuZero("cartpole")
        # >>> muzero.train()
        # >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, restart_wandb_id = None, test=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        self.wandb_run = None
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if restart_wandb_id is not None:
            print(f"Using config from: {restart_wandb_id}")
            wandb_config = get_wandb_config(self.config.wandb_entity, self.config.wandb_project_name, restart_wandb_id)
            self.overwrite_config(wandb_config)

        if config:
            self.overwrite_config(config)

        refresh(self.config)
        print(f"Config: {print_config(self.config)}")

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                    if self.config.max_num_gpus is not None
                    else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True, local_mode=self.config.debug_mode, include_dashboard=False)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "enc_state_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        if self.config.network == "double":
            self.checkpoint["trans_value_loss"] = 0
            self.checkpoint["trans_reward_loss"] = 0
            self.checkpoint["trans_policy_loss"] = 0


        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None


    def overwrite_config(self, config):

        if type(config) is str:
            print(f"Config is string")
            config = json.loads(config)
        if type(config) is dict:
            for param, value in config.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)
                else:
                    raise AttributeError(
                        f"Config has no attribute '{param}'. Check the config file for the complete list of parameters."
                    )


    def init_wandb(self, args):
        if args.wandb_run_id is not None: # restart wandb run
            self.wandb_run = wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project_name,
                resume="must",
                dir=str(self.config.results_path),
                id=args.wandb_run_id
            )

        else:
            self.wandb_run = wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project_name,
                name=str(self.config.log_name),  # to string:
                config=self.config.__dict__,
                dir=str(self.config.results_path),
                resume="allow",
            )
            import yaml
            fp = f"{str(self.config.results_path)}/config.yaml"
            with open(fp, "w") as f:
                yaml.dump(self.config, f)
            wandb.save(fp)



    def train(self):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        logger = self.config.logger

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + (logger is not None) * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
            self.wandb_run,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        print("num_gpus", self.num_gpus)
        print("num_workers", self.config.num_workers)
        print("num_gpus_per_worker", num_gpus_per_worker)
        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Game,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if logger is not None:
            num_gpus = num_gpus_per_worker if self.config.selfplay_on_gpu else 0
            self.test_worker = self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus,
            ).remote(
                self.checkpoint,
                self.Game,
                self.config,
                self.config.seed + self.config.num_workers,
            )
            self.test_worker.continuous_self_play.remote(
                self.shared_storage_worker, None, True
            )


            logging_loop(self, logger)



    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                        0
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return [history.reward_history for history in results]

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint = torch.load(checkpoint_path)
        print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            print("steps:", self.checkpoint["num_played_steps"])
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


    def get_wandb_artifacts(self, run_id, wandb_model_number = None, download_replay_buffer = False):

        # Initialize W&B API
        api = wandb.Api()

        # Define the entity, project, and run ID
        entity = self.config.wandb_entity
        project = self.config.wandb_project_name

        # Fetch the run
        run = api.run(f"{entity}/{project}/{run_id}")
        artifacts = run.logged_artifacts()

        latest_artifacts = sorted(artifacts, key=lambda a: a.created_at)[-3:]

        if wandb_model_number is not None:
            model_artifact = next(a for a in artifacts if str(wandb_model_number) in a.name)
        else:
            model_artifact = next(a for a in latest_artifacts if a.type == 'model')

        # Filter for the model artifact and the data artifact
        data_artifact = next(a for a in latest_artifacts if a.type == 'data')

        model_path = model_artifact.download() + "/model.checkpoint"
        buffer_path = None
        if download_replay_buffer:
            buffer_path = data_artifact.download() + "/replay_buffer.pkl"


        return model_path, buffer_path



@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = mz_net.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests, logger="tensorboard"
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train()
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train()
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / game_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        replay_buffer_path = options[choice] / "replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )

def cmd_line_init():
    games = [
        filename.stem
        for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
        if filename.name != "abstract_game.py"
    ]
    for i in range(len(games)):
        print(f"{i + 1}. {games[i]}")
    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i + 1) for i in range(len(games))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    # Initialize MuZero
    choice = int(choice)
    game_name = games[choice]
    print(f"Selected game: {game_name}")
    muzero = MuZero(game_name)

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Diagnose model",
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Hyperparameter search",
            "Debug",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        valid_inputs = [str(i) for i in range(len(options))]
        choice = input("Choose how to run")

        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        if choice == 0:
            muzero.train()
        elif choice == 1:
            load_model_menu(muzero, game_name)
        elif choice == 2:
            muzero.diagnose_model(30)
        elif choice == 3:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 4:
            muzero.test(render=True, opponent="human", muzero_player=0)
        elif choice == 5:
            env = muzero.Game()
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                env.render()
        elif choice == 6:
            # Define here the parameters to tune
            # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
            muzero.terminate_workers()
            del muzero
            budget = 20
            parallel_experiments = 2
            lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
            discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
            parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
            best_hyperparameters = hyperparameter_search(
                game_name, parametrization, budget, parallel_experiments, 20
            )
            muzero = MuZero(game_name, best_hyperparameters)
        elif choice == 7:
            print("TODO implement debug") # TODO
            pass
        else:
            break
        print("\nDone")


def seq_testing(muzero, file, results_file):
    from self_play import update_pred_dict

    model = mz_net.MuZeroNetwork(muzero.config)
    try:
        model.set_weights(muzero.checkpoint["weights"])
    except Exception as e:
        #print(f"Error: {e}")
        print(f"\ntrying new weights\n")
        model.set_weights(SelfPlay.remove_module_prefix(muzero.checkpoint["weights"]))
    model.eval()

    all_runs_dicts = []

    with open(file, 'r') as f:
        preds = [json.loads(line) for line in f.readlines()]
    preds = [pred['results'] for pred in preds]

    def get_obs_as_pair(all_runs):
        def get_single_pair(run):
            observations = [pred['observation'] for pred in run]
            predictions = [pred['predictions'] for pred in run]
            actions_sequences = [[pred['as'] for pred in preds] for preds in predictions]
            return [{"obs": obs, "as": as_seq} for obs, as_seq in zip(observations, actions_sequences)]

        return [get_single_pair(run) for run in all_runs]

    all_runs = get_obs_as_pair(preds)
    for i, run in enumerate(all_runs):
        print(f"Run {i}")
        run_dict = {
            "game": 0,
            "results": []
        }

        for step in run:
            obs_ar = step['obs']
            pred_dict = {
                "observation": obs_ar,
                "predictions": []
            }

            obs = (
                torch.tensor(obs_ar)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            ).unsqueeze(0)

            (
                value,
                reward,
                policy_logits,
                encoded_state,
            ) = model.initial_inference(obs)
            value = models.support_to_scalar(value, muzero.config.support_size).item()
            reward = models.support_to_scalar(reward, muzero.config.support_size).item()

            update_pred_dict(pred_dict, value, reward, policy_logits, [], [0,1,2])

            for actions in step['as']:
                if actions == []:
                    continue
                policy_logits, value, reward = model.prediction(encoded_state, torch.tensor([actions]))
                value = models.support_to_scalar(value, muzero.config.support_size).item()
                reward = models.support_to_scalar(reward, muzero.config.support_size).item()

                update_pred_dict(pred_dict, value, reward, policy_logits, actions,[0,1,2])

            run_dict['results'].append(pred_dict)
        all_runs_dicts.append(run_dict)


    with open(results_file, "w") as f:
        for game_dict in all_runs_dicts:
            json.dump(game_dict, f)
            f.write('\n')  # Add a newline after each JSON object


    return all_runs_dicts



def visualize_model(muzero):
    from torchviz import make_dot
    network = "double"
    muzero.config.network = network

    model = mz_net.MuZeroNetwork(muzero.config)

    batch_size = 1
    observation = torch.randn(batch_size, *muzero.config.observation_shape)  # Example observation
    action = torch.tensor([[0]])
    _,_,_, hidden_state = model.initial_inference(observation)

    if network == "double":
        org_hs = hidden_state
        hidden_state, trans_hidden_state = torch.chunk(hidden_state, 2, dim=0)

    if network == "fully_connected":
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action)
    else:
        value, reward, policy_logits, _ = model.recurrent_inference(hidden_state, action, hidden_state, action)

    if network == "double":
        hidden_state = org_hs

    dot_representation = make_dot((value, reward, policy_logits, hidden_state), params=dict(model.named_parameters()))
    dot_representation.render(f"graphs/representation_graph_{network}_shared", format="png")

def setup_testing(muzero, args):
    if args.model_path is not None:
        muzero.load_model(checkpoint_path=args.model_path)

    if args.test_mode == "seq":
        print("seq testing")
        seq_file = "predictions/preds/5x5_res_ra_1000_old.json"
        results_file = "predictions/double_preds/5x5_trans_on_res_ra_1000.json"
        seq_testing(muzero, seq_file, results_file)

    elif args.test_mode == "viz":
        print("vizualizing")
        visualize_model(muzero)
    elif args.test_mode == "n_maps":
        print("more map testing")
        muzero.config.show_preds = True
        name = "5x5_res_ra_1000_new.json"
        muzero.config.preds_file = f"predictions/preds/{name}"

        #for i in range(3):
        results = muzero.test(render=False, opponent="self", muzero_player=None, num_tests=1000)
        # put results into file
        with open(f"predictions/results/{name}", "w") as f:
            json.dump(results, f)

    else:
        results = muzero.test(render=True, opponent="self", muzero_player=None)
        print(results)
        print(f"total reward: {sum(results[0])}")


def get_wandb_config(entity, project, run_id):
    config = wandb.Api().run(f"{entity}/{project}/{run_id}").config
    # remove "project" attribute
    del config["project"]
    del config["results_path"]
    del config["testing"]
    if "max_time_minutes" in config:
        del config["max_time_minutes"]
    if "observation_shape" in config:
        config["observation_shape"] = tuple(config["observation_shape"])

    return config


def main(args):
    if args.game_name is None:
        cmd_line_init()
        return

    test_mode = args.test_mode

    print(f"Selected game: {args.game_name}")
    muzero = MuZero(args.game_name, args.config, restart_wandb_id=args.wandb_run_id, test=test_mode is not None)
    logger = muzero.config.logger

    if logger:
        if muzero.config.save_model:
            if type(muzero.config.results_path) is str:
                muzero.config.results_path = pathlib.Path(muzero.config.results_path)
            muzero.config.results_path.mkdir(parents=True, exist_ok=True)

        if logger == "wandb":
            muzero.init_wandb(args)

    if args.wandb_run_id is not None: # if wandb id in args
        checkpoint_path, replay_buffer_path = muzero.get_wandb_artifacts(args.wandb_run_id, args.wandb_model_number, download_replay_buffer=test_mode is not None)
    else:
        checkpoint_path = args.model_path
        replay_buffer_path = args.replay_buffer_path

    if checkpoint_path is not None:
        muzero.load_model(checkpoint_path, replay_buffer_path)

    if args.test_mode is not None:
        return setup_testing(muzero, args)
    else:
        muzero.train()


def setup(test=False):
    parser = argparse.ArgumentParser(description="Process config file.")
    parser.add_argument('-c', '--config', type=str, default=None, help='(part of) config as dict')
    parser.add_argument('-tm', '--test_mode', type=str, default=None, help='How to test') # seq or other
    parser.add_argument('-sf', '--seq_file', type=str, default=None, help='If seq testing, load from this file')
    parser.add_argument('-rfc', '--run_from_cluster', type=str, default=None, help='From which cluster to run, none if local')
    parser.add_argument('-game', '--game_name', type=str, default=None, help='Name of the game module')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to model.checkpoint')
    parser.add_argument('-rbp', '--replay_buffer_path', type=str, default=None, help='Path to replay_buffer.pkl')
    parser.add_argument('-wid', '--wandb_run_id', type=str, default=None, help='Wandb id')
    parser.add_argument('-wmn', '--wandb_model_number', type=bool, default=None, help='Model number from wandb id')
    args = parser.parse_args()

    if args.run_from_cluster == "db" or args.run_from_cluster == "rp":
        wandb_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_key, relogin=True)
    elif args.run_from_cluster is None:
        # manual override
        args.game_name = "custom_grid" #"lunarlander_org" # "custom_grid" # #"custom_grid"  # #"gridworld" # #
        args.config = {
            "debug_mode": False or (sys.gettrace() is not None),
        }
        if test or args.config["debug_mode"]:
            args.config["logger"] = None
            args.config["num_workers"] = 1
        # todo cleanup

        #args.wandb_run_id = "6k4ghx4k"
        #args.wandb_model_number = 320000
        #args.model_path = "models/trans/trans_lunar_320k.checkpoint"

        if test:
            args.test_mode = "n_maps!!" #
            args.config={"testing": True}

        logger = "wandb"

        if logger == "tensorboard":
            from tensorboard import program
            log_dir = "./results"  # Your log directory path
            port = 6006

            # Initialize TensorBoard programmatically
            tb = program.TensorBoard()
            tb.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
            url = tb.launch()
        elif logger == "wandb":
            with open("wandb_api_key") as f:
                wandb_key = f.readline()
            wandb.login(key=wandb_key, relogin=True)

    if args.config is not None and type(args.config) is str and args.config.endswith(".json"):
        print("Getting config from file")
        with open(args.config) as f:
            args.config = json.load(f)
        # if it has observation shape
        if "observation_shape" in args.config:
            args.config["observation_shape"] = tuple(args.config["observation_shape"]) # todo clean up

    print(args)
    return args


if __name__ == "__main__":
    args = setup(test=False)
    main(args)
    wandb.finish()
    ray.shutdown()


