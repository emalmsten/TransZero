import copy
import importlib
import json
import math
import pathlib
import pickle
import os

# todo rename models
from trans_zero.utils import models

import numpy
import ray
import torch

from trans_zero.analysis.diagnose_model import DiagnoseModel
from . import replay_buffer, self_play, shared_storage, trainer
import wandb
from trans_zero.utils.config_utils import refresh, print_config, init_config

from trans_zero.utils.muzero_logger import logging_loop, get_initial_checkpoint
from trans_zero.utils.ray_utils import calc_num_gpus, CPUActor



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
            game_module = importlib.import_module("trans_zero.games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        self.config = init_config(self.config, config, restart_wandb_id)
        print(f"Config: {print_config(self.config)}")

        # Fix random generator seed
        # todo make sure the seeds are used everywhere
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.num_gpus, total_gpus = calc_num_gpus(self.config, split_resources_in)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True, local_mode=self.config.debug_mode, include_dashboard=False)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = get_initial_checkpoint(self.config)


        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        self.replay_buffer = {}

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None


    def init_wandb(self, args):
        path = self.config.results_path
        os.makedirs(path, exist_ok=True)

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
        dm = DiagnoseModel(self.checkpoint, self.config)
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


