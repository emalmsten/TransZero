import pandas as pd
import wandb
import os

def rename_subdirectories(directory):
    for subdir in os.listdir(directory):
        # Construct the full path of the subdirectory
        subdir_path = os.path.join(directory, subdir)

        # Ensure it is a directory
        if os.path.isdir(subdir_path):
            # Split the subdirectory name into words (assuming snake_case)
            words = subdir.split('_')

            # Remove the third last word if there are at least 3 words
            if len(words) >= 3:
                del words[-3]

            # Reconstruct the name
            new_name = '_'.join(words)
            new_path = os.path.join(directory, new_name)

            # Rename the subdirectory
            os.rename(subdir_path, new_path)
            print(f"Renamed: {subdir} -> {new_name}")


def download_wandb_data(project, run_type):
    api = wandb.Api()

    for run in api.runs(project):
        # remove the timestamp from the run name
        run_name_words = run.name.split('_')
        del run_name_words [-4:-2]
        run_name = '_'.join(run_name_words)

        run_dir = f"wandb_data/{run_type}/{run_name}"
        if "resnet" in run_name:
            print(f"Skipping {run_name}")
            continue

        print(f"Downloading data for run {run_name}")
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


# Project is specified by <entity/project-name>
project = "elhmalmsten-tu-delft/TransZero"
run_type = "base_run_fulcon_20241119"
#download_wandb_data(project, run_type)
#rename_subdirectories(f"wandb_data/{run_type}")