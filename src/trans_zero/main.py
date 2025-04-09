from trans_zero.analysis.diagnose_model import diagnose_model
from trans_zero.core.muzero import MuZero
import argparse
import json
import os
import pathlib
import sys
import wandb
import ray

import nevergrad


from trans_zero.analysis.hyperparameter_search import hyperparameter_search
from trans_zero.analysis.testing import setup_testing
from trans_zero.utils.muzero_logger import init_wandb, get_wandb_artifacts


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / game_name).glob("*/")
    ) # todo consider if this is the right path
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
            diagnose_model(muzero, horizon=30)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Process config file.")
    parser.add_argument('-c', '--config', type=str, default=None, help='Config as dict or JSON file')
    parser.add_argument('-tm', '--test_mode', type=str, default=None, help='How to test (e.g., seq, n_maps)')
    parser.add_argument('-sf', '--seq_file', type=str, default=None, help='Sequence file for seq testing')
    parser.add_argument('-rfc', '--run_from_cluster', type=str, default=None, help='Cluster identifier')
    parser.add_argument('-game', '--game_name', type=str, default=None, help='Name of the game module')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('-rbp', '--replay_buffer_path', type=str, default=None, help='Path to replay buffer')
    parser.add_argument('-wid', '--wandb_run_id', type=str, default=None, help='Wandb run ID')
    parser.add_argument('-wmn', '--wandb_model_number', type=int, default=None, help='Model number from Wandb')
    return parser.parse_args()


def load_config(config):
    if isinstance(config, str) and config.endswith('.json'):
        with open(config) as f:
            config = json.load(f)

    if 'observation_shape' in config: # todo consider better method
        config['observation_shape'] = tuple(config['observation_shape'])

    return config


def setup_local_args(cmd_args, test):
    cmd_args.game_name = "custom_grid"  # "lunarlander_org" # "custom_grid" # #"custom_grid"  # #"gridworld" #
    # args.wandb_run_id = "6k4ghx4k"
    # args.wandb_model_number = 320000
    # args.model_path = "models/trans/trans_lunar_320k.checkpoint"

    if test:
        cmd_args.test_mode = "n_maps!!"  #
        cmd_args.config = {"testing": True}

    cmd_args.config = {
        "debug_mode": False or (sys.gettrace() is not None),
    }

    # remove logger and use only 1 worker if in debug mode
    if test or cmd_args.config["debug_mode"]:
        cmd_args.config["logger"] = None
        cmd_args.config["num_workers"] = 1


    return cmd_args


def setup(args, test=False):
    if args.run_from_cluster in ("db", "rp"):
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    elif args.run_from_cluster == "kaggle":
        pass
    elif args.run_from_cluster is not None:
        raise ValueError(f"Unknown cluster: {args.run_from_cluster}")
    else:
        args = setup_local_args(args, test)

    if args.config:
        args.config = load_config(args.config)

    print(f"Final args: {args}")
    return args


def main(args):
    if args.game_name is None:
        cmd_line_init()
        return

    print(f"Selected game: {args.game_name}")
    muzero = MuZero(args.game_name, args.config, restart_wandb_id=args.wandb_run_id, test=args.test_mode is not None)

    if muzero.config.logger == "wandb":
        muzero.wandb_run = init_wandb(muzero.config, args)

    if args.wandb_run_id:
        checkpoint_path, replay_buffer_path = get_wandb_artifacts(
            muzero.config, args.wandb_run_id, args.wandb_model_number, download_replay_buffer=args.test_mode is not None
        )
    else:
        checkpoint_path, replay_buffer_path = args.model_path, args.replay_buffer_path

    if checkpoint_path:
        muzero.load_model(checkpoint_path, replay_buffer_path)

    if args.test_mode:
        setup_testing(muzero, args)
    else:
        muzero.train()


if __name__ == "__main__":
    args = parse_args()
    args = setup(args, test=False)

    try:
        main(args)
    finally:
        wandb.finish()
        ray.shutdown()




# todo consider if tensorboard is necessary
# if logger == "tensorboard":
#     from tensorboard import program
#
#     log_dir = "./results"  # Your log directory path
#     port = 6006
#
#     # Initialize TensorBoard programmatically
#     tb = program.TensorBoard()
#     tb.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
#     url = tb.launch()