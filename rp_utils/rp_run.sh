
# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# refresh bashrc
source ~/.bashrc

# make conda env called tz
conda create -n tz python=3.10 -y
conda activate tz

# for lunarlander
apt install swig -y

# install dependencies
#conda env create -f environment.yml
# pytorch torchvision torchaudio pytorch-cuda=12.1
conda install numpy==1.26.4 gymnasium nevergrad=1.0.1 bayesian-optimization=1.2.0 -c conda-forge -y
conda install -c conda-forge "ray-default" box2d-py -y
pip3 install torch torchvision torchaudio
pip install nevergrad wandb minigrid
pip install --upgrade ray

# lunarlander
# npm?
python muzero.py -rfc rp -game "custom_grid"