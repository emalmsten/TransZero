import pandas as pd
import wandb
import os


def download_wandb_data(project, run_type):
    api = wandb.Api()

    for run in api.runs(project):
        # Create a directory for the run's data
        run_dir = f"wandb_data/{run_type}/{run.name}"
        os.makedirs(run_dir, exist_ok=True)

        # Download the run's history
        history = run.history()
        history.to_csv(os.path.join(run_dir, "history.csv"))

        # Download the run's summary
        summary = run.summary._json_dict
        pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "summary.csv"))

        # Download the run's config
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        pd.DataFrame([config]).to_csv(os.path.join(run_dir, "config.csv"))


        for artifact in run.logged_artifacts():
            if "model" in artifact.name and "10000" in artifact.name:  # Check if "model" is in the artifact name
                artifact.download(root=run_dir)
                break

        print(f"Downloaded data for run {run.name}")

# Project is specified by <entity/project-name>
project = "elhmalmsten-tu-delft/TransZero"
run_type = "base_run_resnet"
download_wandb_data(project, run_type)