import datetime
import pathlib

import gym
import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 42
        self.max_num_gpus = 1

        ### Game
        # Cliff Walking observation is a single integer representing the agent's position in the grid
        self.observation_shape = (1, 1, 1)  # Changed to (1, 1, 1) for a single integer observation
        self.action_space = list(range(4))  # 4 possible actions (0: left, 1: down, 2: right, 3: up)
        self.players = list(range(1))
        self.stacked_observations = 0

        self.muzero_player = 0
        self.opponent = None

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 100  # Reduced max moves to 100 for Cliff Walking
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"
        self.support_size = 10

        self.downsample = False
        self.blocks = 1
        self.channels = 2
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 2
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []

        self.encoding_size = 8
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 500
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

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
    Game wrapper for Cliff Walking.
    """

    def __init__(self, seed=None):
        # Changed environment to Cliff Walking
        self.env = gym.make("CliffWalking-v0")
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
        # Updated for Cliff Walking: observation is a single integer
        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[[observation]]]), reward, done  # Observation wrapped to match shape (1, 1, 1)

    def legal_actions(self):
        """
        Returns the legal actions for Cliff Walking.
        """
        return list(range(4))  # 4 possible actions (left, down, right, up)

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
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
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        # Updated action descriptions for Cliff Walking
        actions = {
            0: "Move left",
            1: "Move down",
            2: "Move right",
            3: "Move up",
        }
        return f"{action_number}. {actions[action_number]}"
