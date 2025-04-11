import time
import wandb
import ray
import os

keys = [
    "total_reward",
    "muzero_reward",
    "opponent_reward",
    "episode_length",
    "mean_value",
    "training_step",
    "lr",
    "total_loss",
    "value_loss",
    "reward_loss",
    "policy_loss",
    "enc_state_loss",
    "num_played_games",
    "num_played_steps",
    "num_reanalysed_games",
]

# todo, consider less duplication
def get_initial_checkpoint(config):
    init_checkpoint = {
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
    if config.network == "double":
        init_checkpoint["trans_value_loss"] = 0
        init_checkpoint["trans_reward_loss"] = 0
        init_checkpoint["trans_policy_loss"] = 0

    return init_checkpoint


def init_wandb(config, args):
    print("Training...\nGo to https://wandb.ai/ to see real-time training performance.\n")

    path = config.results_path
    os.makedirs(path, exist_ok=True)

    if args.wandb_run_id is not None: # restart wandb run
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project_name,
            resume="must",
            dir=str(config.results_path),
            id=args.wandb_run_id
        )

    else:
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project_name,
            name=str(config.log_name),  # to string:
            config=config.__dict__,
            dir=str(config.results_path),
            resume="allow",
        )
        import yaml
        fp = f"{str(config.results_path)}/config.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        wandb.save(fp)

    return wandb_run


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



def get_wandb_artifacts(config, run_id, wandb_model_number = None, download_replay_buffer = False):

    # Initialize W&B API
    api = wandb.Api()

    # Define the entity, project, and run ID
    entity = config.wandb_entity
    project = config.wandb_project_name

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



def tensorboard_logging(info, counter, writer):
    writer.add_scalar(
        "1.Total_reward/1.Total_reward",
        info["total_reward"],
        counter,
    )
    writer.add_scalar(
        "1.Total_reward/2.Mean_value",
        info["mean_value"],
        counter,
    )
    writer.add_scalar(
        "1.Total_reward/3.Episode_length",
        info["episode_length"],
        counter,
    )
    writer.add_scalar(
        "1.Total_reward/4.MuZero_reward",
        info["muzero_reward"],
        counter,
    )
    writer.add_scalar(
        "1.Total_reward/5.Opponent_reward",
        info["opponent_reward"],
        counter,
    )
    writer.add_scalar(
        "2.Workers/1.Self_played_games",
        info["num_played_games"],
        counter,
    )
    writer.add_scalar(
        "2.Workers/2.Training_steps", info["training_step"], counter
    )
    writer.add_scalar(
        "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
    )
    writer.add_scalar(
        "2.Workers/4.Reanalysed_games",
        info["num_reanalysed_games"],
        counter,
    )
    writer.add_scalar(
        "2.Workers/5.Training_steps_per_self_played_step_ratio",
        info["training_step"] / max(1, info["num_played_steps"]),
        counter,
    )
    writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
    writer.add_scalar(
        "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
    )
    writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
    writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
    writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)

def wandb_logging(info, counter, offline_cache, is_double_network):
    metrics = {
        "Total_reward": info["total_reward"],
        "Mean_value": info["mean_value"],
        "Episode_length": info["episode_length"],
        "MuZero_reward": info["muzero_reward"],
        "Opponent_reward": info["opponent_reward"],
        "Self_played_games": info["num_played_games"],
        "Training_steps": info["training_step"],
        "Self_played_steps": info["num_played_steps"],
        "Reanalysed_games": info["num_reanalysed_games"],
        "Training_steps_per_self_played_step_ratio": info["training_step"] / max(1,
                                                                                 info["num_played_steps"]),
        "Learning_rate": info["lr"],
        "Total_weighted_loss": info["total_loss"],
        "Value_loss": info["value_loss"],
        "Reward_loss": info["reward_loss"],
        "Policy_loss": info["policy_loss"],
        "Enc_state_loss": info["enc_state_loss"],
    }

    if is_double_network:
        metrics["Trans_value_loss"] = info["trans_value_loss"]
        metrics["Trans_reward_loss"] = info["trans_reward_loss"]
        metrics["Trans_policy_loss"] = info["trans_policy_loss"]

    log_data = {"step": counter, "metrics": metrics}  # Example datametrics, step=counter

    try:
        wandb.log(log_data["metrics"], step = log_data["step"])  # Try logging
        if offline_cache:  # If there are cached logs, sync them
            print("Syncing cached logs...")
            for cached_data in offline_cache:
                wandb.log(cached_data["metrics"], step = cached_data["step"])
            offline_cache.clear()  # Clear cache after syncing

    except:  # If WANDB is offline, cache logs
        print("WANDB connection lost, caching log...")
        offline_cache.append(log_data)  # Store logs locally




def logging_loop(muzero, logger):
    if logger == "tensorboard":
        writer = init_tensorboard(muzero)
    elif logger == "wandb":
        writer = None

    is_double_network = muzero.config.network == "double"

    if is_double_network:
        keys.append("trans_value_loss")
        keys.append("trans_reward_loss")
        keys.append("trans_policy_loss")

    # Loop for updating the training performance
    first_step = muzero.wandb_run.step
    counter = first_step
    offline_cache = []  # Stores logs when offline

    start_time = time.time()  # Record the start time
    # check if config has max time minutes
    max_time_seconds = muzero.config.max_time_minutes * 60 if muzero.config.max_time_minutes else float("inf")

    try:
        while counter == first_step or info[muzero.config.stopping_criterion] < muzero.config.training_steps:

            if time.time() - start_time > max_time_seconds:
                print(f"Max time reached: {muzero.config.max_time_minutes} minutes")
                break

            info = ray.get(muzero.shared_storage_worker.get_info.remote(keys))
            if logger == "tensorboard":
                tensorboard_logging(info, counter, writer)
            elif logger == "wandb":
                wandb_logging(info, counter, offline_cache, is_double_network)
            if counter % 10 == 0 or counter < 3:
                ts_text = f"{info['training_step']}"
                env_text = f"{info['num_played_steps']}"
                if muzero.config.stopping_criterion == "training_step":
                    ts_text += f"/{muzero.config.training_steps}"
                elif muzero.config.stopping_criterion == "num_played_steps":
                    env_text += f"/{muzero.config.training_steps}"
                print(
                    f"Last test reward: {info['total_reward']:.2f} | "
                    f"Training step: {ts_text} | "
                    f"Games: {info['num_played_games']} | "
                    f"Steps: {env_text} | "
                    f"Loss: {info['total_loss']:.2f}"
                )

            # if logger == "wandb" and counter % muzero.config.checkpoint_interval == 0 and counter > 0:
            #     # save_model(muzero)
            #     save_buffer(muzero)

            counter += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        end_script(muzero, writer, logger)

    end_script(muzero, writer, logger)


def end_script(muzero, writer, logger):
    muzero.terminate_workers()

    # if muzero.config.save_model:
    #     # Persist replay buffer to disk
    #     save_buffer(muzero)
    #     # Persist model to disk
    #     # save_model(muzero)

    if logger == "tensorboard":
        writer.close()
    elif logger == "wandb":
        wandb.finish()




def init_tensorboard(muzero):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(muzero.config.results_path)
    hp_table = [
        f"| {key} | {value} |" for key, value in muzero.config.__dict__.items()
    ]
    writer.add_text(
        "Hyperparameters",
        "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    )
    writer.add_text(
        "Model summary",
        muzero.summary,
    )
    print("Training...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n")
    return writer

