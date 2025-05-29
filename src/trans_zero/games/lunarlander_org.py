import gymnasium as gym

import datetime
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame

try:
    import pygame
except ModuleNotFoundError:
    print("pygame is not installed, please install it to use the rendering feature")


class MuZeroConfig:
    def __init__(self, root=None):
        self.root = root or pathlib.Path(__file__).resolve().parents[1]
        cuda = torch.cuda.is_available()
        self.use_s0_for_pred = True  # Use the output of the representation network for the first prediction

        self.max_time_minutes = None
        self.stopping_criterion = 'num_played_steps'  # 'num_played_steps' or 'training_step'
        # todo consider renaming
        self.training_steps = 400000  # Total number of training (or env) steps (ie weights update according to a batch)

        self.expansion_strategy = None
        self.expansion_budget = 4 # atleast 1 node needs to be expanded
        self.subtree_layers = 2

        # action selection
        self.action_selection = "visit"  # mvc or std
        self.PUCT_C = 2.5
        self.PUCT_variant = "visit"
        self.mvc_beta = 0.3
        self.policy_target_type = "visit"
        self.test_ucb = False
        self.expand_all_children = True

        # Local
        self.testing = False
        self.debug_mode = False or self.testing
        self.render_mode = None #"rgb_array"  # None, human or rgb_array
        self.gif_name = "test"

        self.logger = "wandb" if not self.debug_mode else None
        self.wandb_project_name = "TransZeroV3"
        self.wandb_entity = "elhmalmsten-tu-delft"

        # Essentials
        self.network = "transformer"
        self.game_name = "lunarlander_org"

        # Naming
        self.project = "TransZeroV3"

        self.append = "_local_" + "lun_test"  # Turn this to True to run a test
        path = self.root / "data/results" / self.game_name / self.network
        self.name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{self.append}'
        self.log_name = f"{self.game_name}_{self.network}_{self.name}"
        self.results_path = path / self.name

        # Saving
        self.save_model = True
        self.save_interval = 10000

        # GPU
        self.selfplay_on_gpu = cuda and not self.debug_mode
        self.train_on_gpu = cuda and not self.debug_mode
        self.reanalyse_on_gpu = cuda and not self.debug_mode

        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 42  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 1, 8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(4))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation
        self.predict_reward = True

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 8  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.max_moves = 700  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.999  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.support_size = 20  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network


        # Transformer
        self.transformer_layers = 4
        self.transformer_heads = 16
        self.transformer_hidden_size = 64
        self.transformer_dropout = 0.0
        self.transformer_mlp_dim = 2048 #2048

        self.max_seq_length = 50
        self.positional_embedding_type = "sinus"
        self.norm_layer = True
        self.use_proj = False
        self.representation_network_type = "mlp"  # "mlp"  # "res", "cnn" or "mlp"
        # if cnn
        self.conv_layers_trans = [
            # (out_channels, kernel_size, stride)
            (32, 1, 1),  # Output: (batch_size, 16, 3, 3)
            (64, 1, 1),
            # (128, 3, 1)# Output: (batch_size, 32, 1, 1)
        ]
        self.fc_layers_trans = [64]
        self.mlp_head_layers = [16]
        self.cum_reward = False
        self.state_size = None  # (1,1,8) #None #(16,3,3) # same as
        self.stable_transformer = False
        self.use_forward_causal_mask = True
        self.get_fast_predictions = True


        # Vision Transformer
        self.representation_network_type = "mlp"  # "res", "cnn" or "mlp"
        self.use_simple_vit = True
        self.vit_heads = 8
        self.vit_depth = 4
        self.vit_patch_size = 1
        self.vit_mlp_dim = 512
        self.vit_dropout = 0.01


        ### Training
        self.checkpoint_interval = 10
        self.batch_size = 128  # 64  # Number of parts of games to train on at each training step
        self.value_loss_weight = 0.5  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.encoding_loss_weight = None  # None for not using this
        self.loss_weight_decay = None  # None for not using

        # Learning rate
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0075  # Initial learning rate
        self.lr_decay_rate = 0.99  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 0.02 * self.training_steps
        self.warmup_steps = 0.025 * self.training_steps if self.network == "transformer" else 0

        ### Replay Buffer
        self.replay_buffer_size = 100000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Special Networks
        self.show_preds = False and self.network == "double"
        self.value_network = "transformer"
        self.policy_network = "transformer"
        self.reward_network = "transformer"

        # Best known ratio for deterministic version: 0.8 --> 0.4 in 250 self played game (self_play_delay = 25 on GTX 1050Ti Max-Q).
        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

        self.use_softmax = True
        self.softmax_limits = [0.25, 0.5, 0.75, 1]  # [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]#
        self.softmax_temps = [0.4, 0.35, 0.15, 0.05]  # res [ 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] #
        self.mvc_softmax_temps = None


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        softmax_temps = self.mvc_softmax_temps if self.action_selection == "mvc" else self.softmax_temps
        if softmax_temps is None:
            return 0.0 # won't be used # todo
        for i, limit in enumerate(self.softmax_limits):
            if trained_steps < limit * self.training_steps:
                return softmax_temps[i]

    # todo implement set names and paths
    def set_names_and_paths(self):
        self.root = pathlib.Path(self.root)
        path = self.root / "data/results" / self.game_name / self.network
        name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{self.append}'
        log_name = f"{self.game_name}_{self.network}_{name}"
        results_path = path / name
        return name, log_name, results_path

class Game(AbstractGame):
    """
    Game wrapper for LunarLander-v3.
    """

    def __init__(self, seed=None, config=None):
        # Initialize the LunarLander-v3 environment from gymnasium.
        self.save_gif = config.render_mode == "rgb_array"

        self.env = gym.make("LunarLander-v3", render_mode=config.render_mode)

        if seed is not None:
            # In gymnasium, reset returns a tuple (observation, info)
            self.env.reset(seed=seed)

        if self.save_gif:
            self.gif_path = f"data/gifs/lunarlander_org/{config.gif_name}.gif"
            self.rgb_arr = []

        self.done = False

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action: Action from the action_space to take.

        Returns:
            A tuple of (new observation, reward, done) where:
            - observation is wrapped in a numpy array with shape (1, 1, observation_dim)
            - done is True if the game has ended.
        """
        # For gymnasium, step returns (observation, reward, terminated, truncated, info)
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.done = done
        # Wrap observation in triple-nested list to match shape (1, 1, observation_dim)
        return numpy.array([[observation]]), reward, done

    def legal_actions(self):
        """
        Returns the legal actions for LunarLander.
        """
        # LunarLander discrete version has 4 actions:
        # 0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine.
        return list(range(4))

    def reset(self):
        """
        Reset the game for a new episode.

        Returns:
            The initial observation wrapped in a numpy array.
        """
        # Reset returns (observation, info) in gymnasium.
        observation, info = self.env.reset()
        return numpy.array([[observation]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Render the current game state.
        """
        frame = self.env.render()

        if self.save_gif:
            self.rgb_arr.append(frame)

            # if game is over
            if self.done or len(self.rgb_arr) == 700:
                import imageio
                print(f"Saving gif to {self.gif_path}")
                imageio.mimsave(f"{self.gif_path}", self.rgb_arr, fps=30)
                print("Gif saved.")
        #input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a human-readable string.

        Args:
            action_number: An integer representing an action.

        Returns:
            A string that describes the action.
        """
        actions = {
            0: "Do nothing",
            1: "Fire left engine",
            2: "Fire main engine",
            3: "Fire right engine",
        }
        return f"{action_number}. {actions[action_number]}"
