import torch


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
    norm_layer=False,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    if norm_layer:
        layers.append(torch.nn.LayerNorm(output_size))  # Add normalization here

    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

#
#
#
# ##################################
# ######## Fully Connected #########
# class MuZeroFullyConnectedNetworkSeq(AbstractNetwork):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         action_space_size,
#         encoding_size,
#         fc_reward_layers,
#         fc_value_layers,
#         fc_policy_layers,
#         fc_representation_layers,
#         fc_dynamics_layers,
#         support_size,
#     ):
#         super().__init__()
#         self.action_space_size = action_space_size
#         self.full_support_size = 2 * support_size + 1
#
#         # Representation network
#         input_size = (
#             observation_shape[0]
#             * observation_shape[1]
#             * observation_shape[2]
#             * (stacked_observations + 1)
#             + stacked_observations * observation_shape[1] * observation_shape[2]
#         )
#         self.representation_network = mlp(
#             input_size, fc_representation_layers, encoding_size
#         )
#
#         # Dynamics network
#         self.dynamics_encoded_state_network = mlp(
#             encoding_size + self.action_space_size, fc_dynamics_layers, encoding_size
#         )
#         self.dynamics_reward_network = mlp(
#             encoding_size, fc_reward_layers, self.full_support_size
#         )
#
#         # Prediction networks
#         self.prediction_policy_network = mlp(
#             encoding_size, fc_policy_layers, self.action_space_size
#         )
#         self.prediction_value_network = mlp(
#             encoding_size, fc_value_layers, self.full_support_size
#         )
#
#     def prediction(self, encoded_state):
#         policy_logits = self.prediction_policy_network(encoded_state)
#         value = self.prediction_value_network(encoded_state)
#         return policy_logits, value
#
#     def representation(self, observation):
#         encoded_state = self.representation_network(observation.view(observation.shape[0], -1))
#         # Scale encoded state between [0, 1]
#         min_encoded_state = encoded_state.min(1, keepdim=True)[0]
#         max_encoded_state = encoded_state.max(1, keepdim=True)[0]
#         scale_encoded_state = max_encoded_state - min_encoded_state
#         scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#         encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
#         return encoded_state_normalized
#
#     def dynamics(self, encoded_state, action):
#         # One-hot encode the action
#         action_one_hot = torch.zeros((action.shape[0], self.action_space_size), device=action.device)
#         action_one_hot.scatter_(1, action.long().view(-1, 1), 1.0)
#         x = torch.cat((encoded_state, action_one_hot), dim=1)
#
#         next_encoded_state = self.dynamics_encoded_state_network(x)
#         reward = self.dynamics_reward_network(next_encoded_state)
#
#         # Scale encoded state between [0, 1]
#         min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
#         max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
#         scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
#         scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
#         next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
#
#         return next_encoded_state_normalized, reward
#
#     def initial_inference(self, observation):
#         encoded_state = self.representation(observation)
#         policy_logits, value = self.prediction(encoded_state)
#         # Reward is initialized to zero
#         reward = torch.log(
#             torch.zeros(1, self.full_support_size, device=observation.device)
#             .scatter(1, torch.tensor([[self.full_support_size // 2]], device=observation.device).long(), 1.0)
#             .repeat(len(observation), 1)
#         )
#         return value, reward, policy_logits, encoded_state
#
#     def recurrent_inference(self, encoded_state, action):
#         next_encoded_state, reward = self.dynamics(encoded_state, action)
#         policy_logits, value = self.prediction(next_encoded_state)
#         return value, reward, policy_logits, next_encoded_state
#
#
#
# ###### End Fully Connected #######
# ##################################
#
#
# ##################################
# ############# ResNet #############
#
#
#
#
#
# ########### End ResNet###########
# ##################################
#
# class MuZeroResidualNetworkSeq(AbstractNetwork):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         action_space_size,
#         num_blocks,
#         num_channels,
#         reduced_channels_reward,
#         reduced_channels_value,
#         reduced_channels_policy,
#         fc_reward_layers,
#         fc_value_layers,
#         fc_policy_layers,
#         support_size,
#         downsample,
#     ):
#         super().__init__()
#         self.action_space_size = action_space_size
#         self.full_support_size = 2 * support_size + 1
#
#         block_output_size_reward = (
#             reduced_channels_reward
#             * math.ceil(observation_shape[1] / 16)
#             * math.ceil(observation_shape[2] / 16)
#             if downsample
#             else reduced_channels_reward * observation_shape[1] * observation_shape[2]
#         )
#
#         block_output_size_value = (
#             reduced_channels_value
#             * math.ceil(observation_shape[1] / 16)
#             * math.ceil(observation_shape[2] / 16)
#             if downsample
#             else reduced_channels_value * observation_shape[1] * observation_shape[2]
#         )
#
#         block_output_size_policy = (
#             reduced_channels_policy
#             * math.ceil(observation_shape[1] / 16)
#             * math.ceil(observation_shape[2] / 16)
#             if downsample
#             else reduced_channels_policy * observation_shape[1] * observation_shape[2]
#         )
#
#         # Initialize the networks without DataParallel
#         self.representation_network = RepresentationNetwork(
#             observation_shape,
#             stacked_observations,
#             num_blocks,
#             num_channels,
#             downsample,
#         )
#
#         self.dynamics_network = DynamicsNetwork(
#             num_blocks,
#             num_channels + 1,
#             reduced_channels_reward,
#             fc_reward_layers,
#             self.full_support_size,
#             block_output_size_reward,
#         )
#
#         self.prediction_network = PredictionNetwork(
#             action_space_size,
#             num_blocks,
#             num_channels,
#             reduced_channels_value,
#             reduced_channels_policy,
#             fc_value_layers,
#             fc_policy_layers,
#             self.full_support_size,
#             block_output_size_value,
#             block_output_size_policy,
#         )
#
#     def prediction(self, encoded_state):
#         policy, value = self.prediction_network(encoded_state)
#         return policy, value
#
#     def representation(self, observation):
#         encoded_state = self.representation_network(observation)
#
#         # Normalize the encoded state
#         min_encoded_state = (
#             encoded_state.view(
#                 -1,
#                 encoded_state.shape[1],
#                 encoded_state.shape[2] * encoded_state.shape[3],
#             )
#             .min(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         max_encoded_state = (
#             encoded_state.view(
#                 -1,
#                 encoded_state.shape[1],
#                 encoded_state.shape[2] * encoded_state.shape[3],
#             )
#             .max(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         scale_encoded_state = max_encoded_state - min_encoded_state
#         scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#         encoded_state_normalized = (
#             encoded_state - min_encoded_state
#         ) / scale_encoded_state
#         return encoded_state_normalized
#
#     def dynamics(self, encoded_state, action):
#         # One-hot encode the action
#         action_one_hot = (
#             torch.ones(
#                 (
#                     encoded_state.shape[0],
#                     1,
#                     encoded_state.shape[2],
#                     encoded_state.shape[3],
#                 ),
#                 device=action.device,
#             ).float()
#         )
#         action_one_hot = action[:, :, None, None] * action_one_hot / self.action_space_size
#         x = torch.cat((encoded_state, action_one_hot), dim=1)
#
#         next_encoded_state, reward = self.dynamics_network(x)
#
#         # Normalize the next encoded state
#         min_next_encoded_state = (
#             next_encoded_state.view(
#                 -1,
#                 next_encoded_state.shape[1],
#                 next_encoded_state.shape[2] * next_encoded_state.shape[3],
#             )
#             .min(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         max_next_encoded_state = (
#             next_encoded_state.view(
#                 -1,
#                 next_encoded_state.shape[1],
#                 next_encoded_state.shape[2] * next_encoded_state.shape[3],
#             )
#             .max(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
#         scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
#         next_encoded_state_normalized = (
#             next_encoded_state - min_next_encoded_state
#         ) / scale_next_encoded_state
#         return next_encoded_state_normalized, reward
#
#     def initial_inference(self, observation):
#         encoded_state = self.representation(observation)
#         policy_logits, value = self.prediction(encoded_state)
#         reward = torch.log(
#             (
#                 torch.zeros(1, self.full_support_size)
#                 .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
#                 .repeat(len(observation), 1)
#                 .to(observation.device)
#             )
#         )
#         return value, reward, policy_logits, encoded_state
#
#     def recurrent_inference(self, encoded_state, action):
#         next_encoded_state, reward = self.dynamics(encoded_state, action)
#         policy_logits, value = self.prediction(next_encoded_state)
#         return value, reward, policy_logits, next_encoded_state
#
#
# ########### End ResNet Seq###########
# ##################################
#
#
