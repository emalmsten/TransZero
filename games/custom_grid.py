import datetime
import pathlib

import gymnasium as gym
import numpy
import torch

from .abstract_game import AbstractGame

maps = {
    "2x2_0h_0d": [
        "SF",
        "FG",
    ],
    "3x3_2h_2d": [
        "SFH",
        "FHF",
        "FFG",
    ],
    "4x4_3h_1d": [
        "SFHF",
        "FFFF",
        "HFFF",
        "FHFG",
    ],
}
min_moves = {
    "2x2_0h_0d": 3,
    "3x3_2h_2d": 6,
    "4x4_3h_1d": 8,
}

try:
    import minigrid
    from minigrid.core.constants import COLOR_NAMES
    from minigrid.core.grid import Grid
    from minigrid.core.mission import MissionSpace
    from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
    from minigrid.manual_control import ManualControl
    from minigrid.minigrid_env import MiniGridEnv


except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install minigrid"')


class MuZeroConfig:
    def __init__(self, root=None):
        self.root = root or pathlib.Path(__file__).resolve().parents[1]
        cuda = torch.cuda.is_available()

        # Local
        self.testing = False
        self.debug_mode = False or self.testing

        # Essentials
        self.network = "resnet"
        self.game_name = "custom_grid"
        self.logger = "wandb" if not self.debug_mode else None
        self.custom_map = "3x3_2h_2d"

        # Naming
        self.append = "_local_" + "grid_test"  # Turn this to True to run a test
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

        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 42  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (7, 7, 3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation
        self.negative_reward = 0

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.max_moves = 12  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25  # 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        # Transformer
        self.transformer_layers = 3
        self.transformer_heads = 4
        self.transformer_hidden_size = 32
        self.max_seq_length = 50
        self.positional_embedding_type = "sinus"
        self.norm_layer = True
        self.use_proj = False

        ### Training
        self.training_steps = 15000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 0.95  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000
        self.warmup_steps = 0.025 * self.training_steps if self.network == "transformer" else 0

        ### Replay Buffer
        self.replay_buffer_size = 15000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on
        self.softmax_limits = [0.25, 0.5, 1.0]
        self.softmax_temps =  [1, 0.5, 0.25]

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        for i, limit in enumerate(self.softmax_limits):
            if trained_steps < limit * self.training_steps:
                return self.softmax_temps[i]
        # if trained_steps < 0.1 * self.training_steps:
        #     return 2
        # if trained_steps < 0.25 * self.training_steps:
        #     return 0.8
        # if trained_steps < 0.5 * self.training_steps:
        #     return 0.4
        # elif trained_steps < 0.75 * self.training_steps:
        #     return 0.2
        # else:
        #     return 0.01


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None, config=None):

        gym.envs.registration.register(
            id="CustomSimpleEnv-v0",
            entry_point=__name__ + ":SimpleEnv",
        )

        self.env = gym.make("CustomSimpleEnv-v0", negative_reward = config.negative_reward,
                            custom_map = config.custom_map, testing = config.testing, max_steps = config.max_moves)

        # if seed is not None:
        #     self.env.seed(seed)

        self.env = minigrid.wrappers.ImgObsWrapper(self.env)



    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _, _ = self.env.step(action)
        return numpy.array(observation), reward * 10, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(3))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, info = self.env.reset()
        return numpy.array(observation)

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
        actions = {
            0: "Turn left",
            1: "Turn right",
            2: "Move forward",
            3: "Pick up an object",
            4: "Drop the object being carried",
            5: "Toggle (open doors, interact with objects)",
        }
        return f"{action_number}. {actions[action_number]}"


# def make_custom_simple_env(size=5, max_steps=None):
#     def _init():
#         return SimpleEnv(size=size, max_steps=max_steps)
#     return _init

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.testing = kwargs.pop("testing", False)
        render_mode = "human" #if self.testing else None

        self.negative_reward = kwargs.pop("negative_reward", None)

        self.custom_map_name = kwargs.pop("custom_map", "3x3_2h_2d")
        self.custom_map = maps[self.custom_map_name]

        self.size = int(self.custom_map_name[0]) + 2
        self.min_actions = min_moves[self.custom_map_name]

        self.max_steps = kwargs.pop("max_steps", 0)
        self.max_steps = self.min_actions * 2


        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=self.max_steps,
            render_mode = render_mode,
            **kwargs,
        )

    def _gen_grid_from_string(self, layout):
        width = len(layout[0]) + 2
        height = len(layout) + 2
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for y, row in enumerate(layout):
            y += 1
            for x, char in enumerate(row):
                x += 1
                if char == 'S':
                    self.agent_pos = (x, y)  # Set agent's starting position
                    self.agent_dir = self.agent_start_dir
                elif char == 'H':
                    self.grid.set(x, y, Lava())
                elif char == 'G':
                    self.put_obj(Goal(), x, y)


    @staticmethod
    def _gen_mission():
        return "custom mission"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self._gen_grid_from_string(self.custom_map)
        self.mission = "custom mission"


    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Add custom reward logic
        if reward > 0.0:
            reward = 0.5 + 0.5 * (1 - (self.step_count - self.min_actions) / (self.max_steps - self.min_actions))

        # if self.step_count < self.max_steps and done and reward < 0.00001:
        #     #reward = self.negative_reward
        #     pass

        return obs, reward, done, truncated, info
