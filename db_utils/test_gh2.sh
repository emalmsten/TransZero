#!/bin/bash
#
#SBATCH --job-name="test_git"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128M
#SBATCH --account=Education-EEMCS-MSc-CS

module load 2024r1
module load miniconda3
module load cuda/12.6

# Project and GitHub details
PROJECT_NAME="TransZero"
BRANCH="delft_blue"
REPO_URL="https://github.com/emalmsten/${PROJECT_NAME}.git"

# Load necessary modules
module load 2023r1
module load cuda/11.6

# Read GitHub token
GITHUB_TOKEN=$(<github_token.txt)

# Create working directory
WORKDIR="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit

# Clone or update the repository
if [ -d "$PROJECT_NAME" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd "$PROJECT_NAME" || exit
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    echo "Cloning repository..."
    git clone -b "$BRANCH" "https://${USER}:${GITHUB_TOKEN}@${REPO_URL}"
    cd "$PROJECT_NAME" || exit
fi

# Check the latest commit
git log -1 --pretty=format:"%h %an %s %ad"

# Run the desired script (replace with the actual script you want to execute)
echo "Running the script..."
srun python muzero.py -c configs/test_config.json -rfc db

# Clean up (optional)
echo "Cleaning up..."
rm -rf "$WORKDIR"
