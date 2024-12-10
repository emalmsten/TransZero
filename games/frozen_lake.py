import datetime
import pathlib
import time

import gymnasium as gym
import numpy
import torch

from .abstract_game import AbstractGame
from utils import reset_names, refresh

### 0d is no holes, 1d is for easy map, 2d is for medium, 3d is for hard
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP
# "SFH",
# "FFF",
# "FGF",
#112
maps = {
    "2x2_0h_0d": [
        "SF",
        "FG",
    ],
    "3x3_0h_0d": [
        "SFF",
        "FFF",
        "FGF",
    ],
    "4x4_0h_0d": [
        "SFFF",
        "FFFF",
        "FFFG",
        "FFFF",
    ],
    "2x2_1h_1d": [
        "SF",
        "HG",
    ],
    "3x3_1h_1d": [
        "SFH",
        "FFF",
        "FGF",
    ],
    "3x3_1h_2d": [
        "SFF",
        "HFF",
        "GFF",
    ],
    "3x3_2h_1d": [
        "SFH",
        "FFF",
        "HFG",
    ],
    "3x3_2h_2d": [
        "SFH",
        "FHF",
        "FFG",
    ],
    "3x3_3h_2d": [
        "SFH",
        "HFF",
        "FHG",
    ],
    "4x4_1h_1d": [
        "SFFF",
        "FFFF",
        "FFFG",
        "FFFH",
    ],
    # y
    "4x4_2h_1d": [
        "SFFF",
        "FFHF",
        "FFFF",
        "FGFH",
    ],
    "4x4_2h_2d": [
        "SFFF",
        "FFFF",
        "FFFH",
        "FHFG",
    ],
    "4x4_3h_1d": [
        "SFHF",
        "FFFF",
        "HFFF",
        "FHFG",
    ],
    "4x4_3h_2d": [
        "SFHF",
        "FFHF",
        "FFFF",
        "FHGF",
    ],
    # almost impossible with slipperiness
    "4x4_5h_3d": [
        "SFFF",
        "HHHF",
        "FFFF",
        "HGHF",
    ],
    "5x5_3h_2d": [
        "SFHFF",
        "FFFFF",
        "FFFFF",
        "FFHFF",
        "HFFGF",
    ]
}


class MuZeroConfig:

    def __init__(self, root=None):
        self.root = root or pathlib.Path(__file__).resolve().parents[1]
        cuda = torch.cuda.is_available()

        # Local
        self.testing = False
        self.debug_mode = False or self.testing

        # Essentials
        self.network = "transformer"
        self.game_name = "frozen_lake"
        self.custom_map = "3x3_1h_1d"
        self.logger = "wandb" if not self.debug_mode else None

        # Naming
        self.append = "_local_" + "speedTest"  # Turn this to True to run a test
        path = self.root / "results" / self.game_name / self.custom_map / self.network
        self.name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{self.append}'
        self.log_name = f"{self.game_name}_{self.custom_map}_{self.network}_{self.name}"
        self.results_path = path / self.name

        # Saving
        self.save_model = True
        self.save_interval = 500

        # GPU
        self.selfplay_on_gpu = cuda and not self.debug_mode
        self.train_on_gpu = cuda and not self.debug_mode
        self.reanalyse_on_gpu = cuda and not self.debug_mode

        # fmt: off
        self.seed = 43
        self.max_num_gpus = 1

        ### Game
        # Frozen Lake observation is a single integer representing the agent's position
        self.observation_shape = (1, 1, 1)  # Changed to (1, 1, 1) for a single integer observation
        self.action_space = list(range(4))  # Updated action space: 4 possible actions (0: left, 1: down, 2: right, 3: up)
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = None

        ### Self-Play
        self.num_workers = 1
        self.max_moves = 50  # Reduced max moves for Frozen Lake
        self.num_simulations = 25
        self.discount = 0.997
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.support_size = 10

        # Residual Network
        self.downsample = False
        self.blocks = 1
        self.channels = 2
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 2
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []

        # Fully Connected
        self.encoding_size = 8
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]

        # Transformer
        self.transformer_layers=2
        self.transformer_heads=2
        self.transformer_hidden_size=16
        self.max_seq_length=50
        self.positional_embedding_type='sinus'  # sinus or learned
        self.norm_layer = False
        self.use_proj = False

        ### Training
        self.checkpoint_interval = 10
        self.training_steps = 10000
        self.batch_size = 128
        self.value_loss_weight = 1

        # Learning Rate
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 0.1 * self.training_steps
        self.warmup_steps = 0.025 * self.training_steps if self.network == "transformer" else 0

        ### Replay Buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Special Networks
        self.show_preds = False and self.network == "double"
        self.value_network = "transformer"
        self.policy_network = "transformer"
        self.reward_network = "transformer"

        ### Other
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5
        # fmt: on


    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper for Frozen Lake.
    """

    def __init__(self, seed=None, config=None):
        # Changed environment to Frozen Lake
        if config is not None:
            print(f"Using custom map: {config.custom_map}")
            custom_map = maps[config.custom_map]
            [print(row) for row in custom_map]
            self.env = gym.make("FrozenLake-v1", is_slippery=False, desc=custom_map,
                                render_mode="human" if config.testing else None)
        else:
            self.env = gym.make("FrozenLake-v1", is_slippery=False)
        if seed is not None:
            self.env.reset(seed=seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # Updated for Frozen Lake: observation is a single integer
        observation, reward, done, truncated, info = self.env.step(action)
        return numpy.array([[[observation]]]), reward, done or truncated  # Observation wrapped to match shape (1, 1, 1)

    def legal_actions(self):
        """
        Returns the legal actions for Frozen Lake.
        """
        return list(range(4))  # Updated: 4 possible actions (left, down, right, up)

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, info = self.env.reset()
        return numpy.array([[[observation]]])  # Observation wrapped to match shape (1, 1, 1)

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        time.sleep(1)
        #input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        # Updated action descriptions for Frozen Lake
        actions = {
            0: "Move left",
            1: "Move down",
            2: "Move right",
            3: "Move up",
        }
        return f"{action_number}. {actions[action_number]}"


    # "3x3_test": [
    #     "FGF",
    #     "HSH",
    #     "FGF",
    # ],