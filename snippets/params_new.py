
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
self.pb_c_init = 1.25  # 1.25

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

self.encoding_size = 8


### Training
self.training_steps = 15000  # Total number of training steps (ie weights update according to a batch)
self.batch_size = 256  # Number of parts of games to train on at each training step
self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)

self.weight_decay = 1e-4  # L2 weights regularization
self.momentum = 0.9  # Used only if optimizer is SGD

# Exponential learning rate schedule
self.lr_init = 0.015  # Initial learning rate
self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
self.lr_decay_steps = 1000
self.warmup_steps = 0.025 * self.training_steps if self.network == "transformer" else 0

### Replay Buffer
self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

### Adjust the self play / training ratio to avoid over/underfitting
self.self_play_delay = 0  # Number of seconds to wait after each played game
self.training_delay = 0  # Number of seconds to wait after each training step
self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
# fmt: on
self.softmax_limits = [0.25, 0.5, 1.0]
self.softmax_temps = [1, 0.5, 0.25]