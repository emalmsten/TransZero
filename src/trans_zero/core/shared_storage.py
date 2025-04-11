import copy

import ray
import torch
import wandb
import pickle


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config, wandb_run=None):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        print("Training step", self.current_checkpoint["training_step"])
        print("Env Step", self.current_checkpoint["num_played_steps"])
        self.wandb_run = wandb_run

    def save_checkpoint(self, num_played_steps, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)
        # save to wandb
        if self.config.logger == "wandb" and self.wandb_run is not None:
            print("saving checkpoint to wandb")
            artifact = wandb.Artifact(
                name=f"model_step-{num_played_steps}_name-{self.config.name}", type="model")
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)

    def save_buffer(self, replay_buffer, num_played_steps, num_played_games, num_reanalysed_games, path=None):
        if not path:
            path = self.config.results_path / "replay_buffer.pkl"

        buffer = ray.get(replay_buffer.get_buffer.remote())
        print("played_steps:", num_played_steps)

        pickle.dump(
            {
                "buffer": buffer,
                "num_played_games": num_played_games,
                "num_played_steps": num_played_steps,
                "num_reanalysed_games": num_reanalysed_games,
            },
            open(path, "wb"),
        )

        if self.wandb_run is not None:
            try:
                previous_artifact = self.wandb_run.use_artifact(f'buffer-{self.config.name}:latest')
                previous_artifact.delete(delete_aliases=True)
                print("Deleted previous artifact")
            except Exception as e:
                print(f"Could not delete previous artifact due to: {e}")

            print("Saving buffer to wandb")
            rb_artifact = wandb.Artifact(name=f'buffer-{self.config.name}', type="data")
            rb_artifact.add_file(str(path))
            try:
                self.wandb_run.log_artifact(rb_artifact, aliases=["latest"])
            except Exception as e:
                print(f"Could not log artifact due to: {e}")

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
