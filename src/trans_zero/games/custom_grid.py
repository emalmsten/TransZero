import datetime
import pathlib
import time

import gymnasium as gym
import numpy as np
import torch

from trans_zero.utils.optimal_path_finder import calculate_steps_and_turns_to_goal
from .abstract_game import AbstractGame
from trans_zero.paths import PROJECT_ROOT


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
    "3x3_2h_3d": [
        "FFF",
        "FHH",
        "FFG",
    ],
    "4x4_3h_1d": [
        "SFHF",
        "FFFF",
        "HFFF",
        "HHFG",
    ],
    "4x4_3h_3d": [
        "SHGH",
        "FHFF",
        "FHHF",
        "FFFF",
    ],
    "5x5_3h_3d": [
        "SFHFG",
        "FFHFF",
        "FFHFF",
        "FFFFF",
        "FFFFF"
    ],
    "5x5_9h_3d": [
        "SHHHG",
        "FHHHF",
        "FHHHF",
        "FFFFF",
        "HFFFH"
    ],
    "6x6_3h_3d": [
        "SFHFGF",
        "FFHFFF",
        "FFFHFF",
        "HFFHFF",
        "HFFFFF",
        "FFFFFF"
    ],
    "6x6_5h_1d": [
        "SFHFFF",
        "FFFFFF",
        "FFFFFF",
        "HFFFFF",
        "HFFFFF",
        "FFFFFG"
    ],
    "5x5_4h_1d": [
        "SFHFF",
        "FFFFF",
        "FFFFF",
        "HFFFF",
        "FFFFG"
    ],
}
max_moves = {
    "2x2_0h_0d": 6,
    "3x3_2h_2d": 15,
    "3x3_2h_3d": 15,

    "4x4_3h_1d": 36,
    "4x4_3h_3d": 18,

    "5x5_3h_3d": 25,
    "5x5_9h_3d": 25,
    "6x6_3h_3d": 25,

    "5x5_4h_1d": 25,
    "6x6_5h_1d": 25
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
        self.root = PROJECT_ROOT if root is None else root
        cuda = torch.cuda.is_available()

        self.max_time_minutes = None
        self.stopping_criterion = 'num_played_steps'  # 'num_played_steps' or 'training_step'
        self.training_steps = 25000  # Total number of training steps (ie weights update according to a batch)

        self.expansion_strategy = None #"deep_new" #"deep" #None #
        self.subtree_layers = 2
        self.expansion_budget = 39 # atleast 1 node needs to be expanded
        self.num_simulations = 25  # Number of future moves self-simulated
        self.max_seq_length = 25 # todo reconsider length
        self.expand_all_children = True

        # action selection
        self.action_selection = "mvc" # mvc or visit
        self.PUCT_C = 2.5
        self.PUCT_variant = "mvc"
        self.mvc_beta = 0.3
        self.policy_target_type = "mvc"
        self.test_ucb = False

        # Local
        self.testing = False
        self.show_preds = False and self.testing     # can only be done in testing
        self.preds_file = f"{self.root}/data/predictions/4x4_preds/transformer/test.json"
        self.debug_mode = False or self.testing

        self.logger = "wandb" if not self.debug_mode else None
        self.wandb_project_name = "TransZeroV3"
        self.wandb_entity = "elhmalmsten-tu-delft"

        # Essentials
        self.network = "transformer"
        self.game_name = "custom_grid"

        self.custom_map = "3x3_2h_2d" #"4x4_3h_3d" # #
        self.start_pos = None #(3,1) #None #(0,1)
        self.start_dir = None # 0: right, 1: down, 2: left, 3: up
        self.random_map = False
        self.pov = '1_hot_god' # agent, god, 1_hot_god, 2_hot_god

        # Naming
        self.append = "_local_" + "grid_test"  # Turn this to True to run a test

        self.name, self.log_name, self.results_path = self.set_names_and_paths()

        # Saving
        self.save_model = True
        self.save_interval = 5000

        # GPU
        self.selfplay_on_gpu = cuda and not self.debug_mode
        self.train_on_gpu = cuda and not self.debug_mode
        self.reanalyse_on_gpu = cuda and not self.debug_mode

        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 42  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = self.get_observation_shape(self.pov) #  # (7, 7, 3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation
        self.negative_reward = -0.0 #-0.1
        self.obstacle = "lava"
        self.predict_reward = True # todo check if this should even be an option

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 2 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.max_moves = max_moves[self.custom_map] # Maximum number of moves if game is not finished before
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3 # 0.25
        self.root_exploration_fraction = 0.3

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = None  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 12  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = [32]  # Define the hidden layers in the representation network

        self.fc_dynamics_layers = [32]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [32]  # Define the hidden layers in the reward network
        self.fc_value_layers = [32]  # Define the hidden layers in the value network
        self.fc_policy_layers = [32]  # Define the hidden layers in the policy network

        # Transformer
        self.transformer_layers = 4
        self.transformer_heads = 8
        self.transformer_hidden_size = 32
        self.positional_embedding_type = "sinus"
        self.norm_layer = True
        self.use_proj = False
        self.representation_network_type = "mlp"  # "res", "cnn" or "mlp"
        # if cnn
        self.conv_layers_trans = [
                # (out_channels, kernel_size, stride)
                (16, 3, 1),  # Output: (batch_size, 16, 3, 3)
                (32, 3, 1),
                # (128, 3, 1)# Output: (batch_size, 32, 1, 1)
            ]
        self.fc_layers_trans = [128, 64]
        self.mlp_head_layers = [32, 16]
        self.cum_reward = False
        self.state_size = None #(16, 4, 4) # same as
        self.stable_transformer = False
        self.get_fast_predictions = True # todo rename
        self.use_forward_causal_mask = True


        ### Training
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.5  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.encoding_loss_weight = None # None for not using this
        self.loss_weight_decay = None # None for not using

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001 # res: 0.015
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 5000 # todo 1000 for res but test if it can just be 5000
        self.warmup_steps = 0.1 * self.training_steps if self.network == "transformer" else 0

        ### Replay Buffer
        self.replay_buffer_size = 150000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
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
        self.use_softmax = True
        self.softmax_limits = [0.25, 0.5, 0.75, 1] # res: 0.25, 0.5, 1
        self.softmax_temps =  [1, 0.75, 0.5, 0.25] # res 1, 0.5, 0.25
        self.mvc_softmax_temps = None #[1, 0.75, 0.5, 0.25] #None #[1, 0.5, 0.25, 0.1]

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        softmax_temps = self.mvc_softmax_temps if self.action_selection == "mvc" else self.softmax_temps
        if softmax_temps is None:
            return 1.0  # won't be used # todo
        for i, limit in enumerate(self.softmax_limits):
            if trained_steps < limit * self.training_steps:
                return softmax_temps[i]


    def get_observation_shape(self, pov):
        size = int((self.custom_map[0]))
        if pov == 'agent':
            return (1, min(7, size+1), 7)
        elif 'god' in pov:
            channels = 1
            if pov == '1_hot_god':
                channels = 6
            elif pov == '2_hot_god':
                channels = 3
            elif pov == 'kinda_god':
                return (1, 1, 3 + 2*(size-1))

            return (channels, int(self.custom_map[0]), int(self.custom_map[2]))
        else:
            raise ValueError('POV must be either "agent" or "god"')


    def get_max_moves(self, custom_map):
        return max_moves[custom_map]


    def set_names_and_paths(self):
        path = pathlib.Path(self.root) / "data/results" / self.game_name / self.custom_map / self.network
        name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}{self.append}'
        log_name = f"{self.game_name}_{self.custom_map}_{'random' if self.random_map else'fixed'}_{self.pov}_{self.network}_{name}"
        results_path = path / name
        return name, log_name, results_path


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None, config=None):

        gym.envs.registration.register(
            id="CustomSimpleEnv-v0",
            entry_point=__name__ + ":SimpleEnv",
        )
        self.config = config
        self.size = int(config.custom_map[0]) + 2

        self.env = gym.make("CustomSimpleEnv-v0",
                            negative_reward = config.negative_reward,
                            custom_map = config.custom_map,
                            testing = config.testing,
                            max_steps = max_moves[config.custom_map],
                            start_pos = config.start_pos,
                            start_dir = config.start_dir,
                            random_map = config.random_map,
                            obstacle = config.obstacle,
                            )

        # todo add seed
        # if seed is not None:
        #     self.env.seed(seed)

        #self.env = minigrid.wrappers.ImgObsWrapper(self.env)
        if 'god' in config.pov:
            self.env = minigrid.wrappers.FullyObsWrapper(self.env)


    @staticmethod
    def one_dim_encode_grid(obs, dir):
        agent_positions = np.argwhere(obs == 10)

        agent_idx = agent_positions[0]
        # Swap indices: column is x, row is y.
        agent_coord = [agent_idx[1], agent_idx[2]]

        # Find all lava positions (value 8)
        lava_indices = np.argwhere(obs == 9)
        # Swap indices for each lava coordinate
        lava_coords = [[pos[1], pos[2]] for pos in lava_indices]

        # Combine agent and lava coordinates into one flat list.
        # The resulting list is: [agent_x, agent_y, lava1_x, lava1_y, lava2_x, lava2_y, ...]
        result = np.array(agent_coord + [coord for pair in lava_coords for coord in pair])

        # prepend dir
        result = np.insert(result, 0, dir)
        # insert two one dim dimensions
        result = np.expand_dims(result, axis=0)
        result = np.expand_dims(result, axis=0)

        return result

    @staticmethod
    def one_hot_encode_grid(obs, dir):
        """
        Converts a 2D observation grid into a multi-channel one-hot encoded representation.

        Args:
        - obs: 2D NumPy array of shape (H, W), representing the world.
        - direction: Integer (0-3) indicating the agent's direction.

        Returns:
        - A NumPy array of shape (C, H, W) where C is the number of channels.
        """
        _, H, W = obs.shape
        num_dirs = 4  # Four possible directions (up, right, down, left)

        # Create empty one-hot encoded map with additional channels
        one_hot_map = np.zeros((num_dirs + 2, H, W), dtype=np.uint8)

        # find agents position
        agent_pos = np.where(obs == 10)

        # Direction encoding
        one_hot_map[dir, agent_pos[1], agent_pos[2]] = 1

        # Lava (8) and Goal (9) encoding
        one_hot_map[4] = (obs == 8).astype(np.uint8)  # Lava channel
        one_hot_map[5] = (obs == 9).astype(np.uint8)  # Goal channel

        return one_hot_map

    @staticmethod
    def two_hot_encode_grid(obs, dir):
        """
        Converts a 2D observation grid into a multi-channel encoded representation.

        Args:
        - obs: 2D NumPy array of shape (H, W), representing the world.
        - direction: Integer (0-3) indicating the agent's direction.

        Returns:
        - A NumPy array of shape (C, H, W) where C is the number of channels.
        """
        _, H, W = obs.shape

        # Create empty one-hot encoded map with additional channels
        one_hot_map = np.zeros((3, H, W), dtype=np.uint8)

        # find agents position
        agent_pos = np.where(obs == 10)

        # Direction encoding
        one_hot_map[0, agent_pos[1], agent_pos[2]] = dir + 1

        # Lava (8) and Goal (9) encoding
        one_hot_map[1] = (obs == 8).astype(np.uint8)  # Lava channel
        one_hot_map[2] = (obs == 9).astype(np.uint8)  # Goal channel

        return one_hot_map


    @staticmethod
    def shape_observation(pov, size, obs):
        """
        Convert the observation to the required shape.

        Args:
            observation : observation from the environment.

        Returns:
            Observation in the required shape.


        """
        if pov == 'agent':
            return obs['image'][:, max(0, (7 - (size - 1))):, [0]].swapaxes(0, 2)
        elif 'god' in pov:
            dir = obs['direction']
            obs = obs['image'].swapaxes(0, 2)[[0], 1:-1, 1:-1]
            #obs = obs.swapaxes(1,2)

            if pov == '1_hot_god':
                return Game.one_hot_encode_grid(obs, dir)
            elif pov == '2_hot_god':
                return Game.two_hot_encode_grid(obs, dir)
            elif pov == 'kinda_god':
                return Game.one_dim_encode_grid(obs, dir)

            # take every 10 and do minus (11 + obs[direction])
            obs = np.where(obs == 10, dir + 3, obs)
            return obs

        else:
            raise ValueError('POV must be either "agent" or "god"')


    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        obs, reward, done, _, _ = self.env.step(action)
        obs = Game.shape_observation(self.config.pov, self.size, obs)

        env = self.env.unwrapped

        # Add custom reward logic
        if reward > 0.0:
            reward = 0.5 + 0.5 * (1 - (env.step_count - env.min_actions) / (env.max_steps - env.min_actions))
            reward = min(1.0, reward)

        if reward < -0.00001:
            print("negative reward")

        if env.step_count < env.max_steps and done and reward < 0.00001:
            reward = env.negative_reward

        #return obs, reward, done #, truncated, info

        return np.array(obs), reward * 10, done

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
        obs, info = self.env.reset()

        #obs = obs[:, max(0, (7-(self.size-1))):, [0]]
        obs = Game.shape_observation(self.config.pov, self.size, obs)

        return np.array(obs)

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        # sleep for 1 second
        time.sleep(0.1)
        #input("Press enter to take a step ")

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
        #agent_start_pos=(1, 1),
        #agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        #self.agent_start_pos = agent_start_pos
        #self.agent_start_dir = agent_start_dir

        self.testing = kwargs.pop("testing", False)
        render_mode = None #"human" #if self.testing else None

        self.custom_map_name = kwargs.pop("custom_map")
        self.random_map = kwargs.pop("random_map")
        self.start_pos = kwargs.pop("start_pos")
        self.start_dir = kwargs.pop("start_dir")
        self.obstacle = kwargs.pop("obstacle")
        self.negative_reward = kwargs.pop("negative_reward")
        self.max_steps = max_steps

        self.custom_map = maps[self.custom_map_name]
        self.size = int(self.custom_map_name[0]) + 2

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=max_steps,
            render_mode = render_mode,
            see_through_walls=True,
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
                if char == 'H':
                    obstacle = Lava() if self.obstacle == "lava" else Wall()
                    self.grid.set(x, y, obstacle)
                elif char == 'S':
                    self.agent_pos = (x, y)
                    self.agent_dir = self.start_dir if self.start_dir is not None else np.random.randint(0, 4)
                elif char == 'G':
                    self.put_obj(Goal(), x, y)


    @staticmethod
    def _gen_mission():
        return "custom mission"

    @staticmethod
    def get_random_map(size, holes):
        # Initialize map with 'F'
        init_map = np.full((size, size), "F", dtype=str)

        # Set the goal position
        init_map[size - 1, size - 1] = "G"

        # Get all valid positions (excluding goal)
        possible_positions = [(x, y) for x in range(size) for y in range(size) if (x, y) != (size - 1, size - 1)]

        # Randomly select positions for 'H'
        selected_positions = np.random.choice(len(possible_positions), holes, replace=False)

        # Assign 'H' to the selected positions
        for idx in selected_positions:
            x, y = possible_positions[idx]
            init_map[y, x] = "H"

        return init_map



    def _gen_grid(self, width, height):
        walkable_size = self.size - 2
        # Create an empty grid
        if self.random_map:
            custom_map = self.get_random_map(walkable_size, int(self.custom_map_name[4]))
        else:
            # turn array (self.custom map) of strings to array of arrays
            custom_map = [list(row) for row in self.custom_map]
            # if there is an S remove it
            custom_map = [["F" if cell == "S" else cell for cell in row] for row in custom_map]


        step_grid = calculate_steps_and_turns_to_goal(custom_map)
        # list all positions which you can reach from the goal
        reachables = [(y, x) for y in range(len(step_grid)) for x in range(len(step_grid[y])) if step_grid[y][x] > 0]

        # if not more than 20% of all positions are reachable, reroll (goal locked between holes)
        if not len(reachables) / walkable_size**2 > 0.2:
            return self._gen_grid(width, height)

        # get random reachable position
        start_pos = self.start_pos
        if start_pos is None:
            start_pos = reachables[np.random.randint(0, len(reachables))]

        # add 3 to the min actions such that the agent can turn in the beginning (todo, not optimal but fine?)
        self.min_actions = step_grid[start_pos[0], start_pos[1]] + 3
        custom_map[start_pos[0]][start_pos[1]] = "S"

        self._gen_grid_from_string(custom_map)
        self.mission = "custom mission"

        if False: # todo
            import json
            file_path = f"data/custom_maps/{walkable_size}x{walkable_size}.json"
            with open(file_path, "a") as f:
                game_dict = {
                    "map": custom_map.tolist(),
                    "start_pos": start_pos,
                    "start_dir": self.agent_dir,
                    "min_actions": int(self.min_actions)
                }
                json.dump(game_dict, f)
                f.write('\n')  # Add a newline after each JSON object
                f.flush()






    #
    # def step(self, action):
    #     if self.step_count == 0:
    #         step_grid = calculate_steps_and_turns_to_goal(self.custom_map)
    #         min_actions = step_grid[self.agent_pos[1] - 1, self.agent_pos[0] - 1]
    #         self.min_actions = min_actions + 2 # inital turning
    #
    #     obs, reward, done, truncated, info = super().step(action)
    #     #obs['image'] = obs['image'][:, max(0, (7-(self.size-1))):, [0]].swapaxes(0, 2)
    #     # if an observation is an 8, turn it into a 4
    #     #obs['image'] = numpy.where(obs['image'] == 8, 4, obs['image'])
    #     obs['image'] = Game.shape_observation(obs['image'])
    #
    #     # Add custom reward logic
    #     if reward > 0.0:
    #         reward = 0.5 + 0.5 * (1 - (self.step_count - self.min_actions) / (self.max_steps - self.min_actions))
    #         reward = min(1.0, reward)
    #
    #     if self.step_count < self.max_steps and done and reward < 0.00001:
    #         reward = self.negative_reward
    #
    #     return obs, reward, done, truncated, info
