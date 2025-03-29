from trans_zero.core.muzero import MuZero, setup_testing, cmd_line_init
import argparse
import json
import os
import pathlib
import sys
import wandb
import ray



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
            pass
            #wandb_key = os.getenv("WANDB_API_KEY")
            #print(wandb_key)
            #wandb.login(key=wandb_key, relogin=True)

    if args.config is not None and type(args.config) is str and args.config.endswith(".json"):
        print("Getting config from file")
        with open(args.config) as f:
            args.config = json.load(f)
        # if it has observation shape
        if "observation_shape" in args.config:
            args.config["observation_shape"] = tuple(args.config["observation_shape"]) # todo clean up

    print(args)
    return args


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

if __name__ == "__main__":
    args = setup(test=False)
    main(args)
    wandb.finish()
    ray.shutdown()