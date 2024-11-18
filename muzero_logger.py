import datetime
import time
import pickle
import wandb
import ray
from torch.utils.tensorboard import SummaryWriter
import torch

# TODO save more checkpoints

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
    "num_played_games",
    "num_played_steps",
    "num_reanalysed_games",
]

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

def wandb_logging(info, counter):
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
    }
    wandb.log(metrics, step=counter)


def logging_loop(muzero, logger):
    if logger == "tensorboard":
        writer = init_tensorboard(muzero)
    elif logger == "wandb":
        init_wandb(muzero)

    # Loop for updating the training performance
    counter = 0
    try:
        while counter == 0 or info["training_step"] < muzero.config.training_steps:
            info = ray.get(muzero.shared_storage_worker.get_info.remote(keys))
            if logger == "tensorboard":
                tensorboard_logging(info, counter, writer)
            elif logger == "wandb":
                wandb_logging(info, counter)
            if counter % 10 == 0 or counter < 3:
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{muzero.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    # end="\r", flush=True,
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

    if muzero.config.save_model:
        # Persist replay buffer to disk
        save_buffer(muzero)
        # Persist model to disk
        # save_model(muzero)

    if logger == "tensorboard":
        writer.close()
    elif logger == "wandb":
        wandb.finish()


# def save_model(muzero):
#     path = muzero.config.results_path / "model.ckpt"
#     print(f"\n\nPersisting model to disk at {path}")
#     torch.save(muzero.state_dict(), str(path))


def save_buffer(muzero):
    path = muzero.config.results_path / "replay_buffer.pkl"
    print(f"\n\nPersisting replay buffer games to disk at {path}, num steps = {muzero.checkpoint['num_played_steps']}")
    pickle.dump(
        {
            "buffer": muzero.replay_buffer,
            "num_played_games": muzero.checkpoint["num_played_games"],
            "num_played_steps": muzero.checkpoint["num_played_steps"],
            "num_reanalysed_games": muzero.checkpoint["num_reanalysed_games"],
        },
        open(path, "wb"),
    )


def init_tensorboard(muzero):
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


def init_wandb(muzero):
    wandb.config.update(muzero.config.__dict__)
    print("Training...\nGo to https://wandb.ai/ to see real-time training performance.\n")
