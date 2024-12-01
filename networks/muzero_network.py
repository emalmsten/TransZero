from networks.fully_connected import MuZeroFullyConnectedNetwork
from networks.resnet import MuZeroResidualNetwork
from networks.transformer import MuZeroTransformerNetwork
from networks.mixed_network import MuZeroMixedNetwork
from networks.double_network import MuZeroDoubleNetwork
from networks.double_network_new import MuZeroNewDoubleNetwork


class MuZeroNetwork:
    def __new__(cls, config):
        debug_mode = hasattr(config, "debug_mode") and config.debug_mode

        if config.network == "fully_connected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
                debug_mode,
            )


        elif config.network == "transformer":
            return MuZeroTransformerNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,

                config.fc_representation_layers,
                config.support_size,

                config.transformer_layers,
                config.transformer_heads,
                config.transformer_hidden_size,
                config.max_seq_length,
                config.positional_embedding_type,  # sinus or learned

                debug_mode,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
                debug_mode
            )
        elif "trans" in config.network:
            return MuZeroMixedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,

                config.transformer_layers,
                config.transformer_heads,
                config.transformer_hidden_size,
                config.max_seq_length,
                config.positional_embedding_type,  # sinus or learned

                config.value_network,
                config.policy_network,
                config.reward_network,

                debug_mode,
            )
        elif config.network == "double":
            return MuZeroDoubleNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,

                config.transformer_layers,
                config.transformer_heads,
                config.transformer_hidden_size,
                config.max_seq_length,
                config.positional_embedding_type,  # sinus or learned

                config.value_network,
                config.policy_network,
                config.reward_network,

                debug_mode,
            )
        elif config.network == "double_new":
            trans_network = MuZeroTransformerNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,

                config.fc_representation_layers,
                config.support_size,

                config.transformer_layers,
                config.transformer_heads,
                config.transformer_hidden_size,
                config.max_seq_length,
                config.positional_embedding_type,  # sinus or learned

                debug_mode,
            )
            fully_network = MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
                debug_mode,
            )
            trans_network.representation_network = trans_network.representation_network

            return MuZeroNewDoubleNetwork(
                trans_network,
                fully_network,
            )


        else:
            raise NotImplementedError(
                f'The network parameter should be "transformer", "fully_connected" or "resnet". Received: {config.network}'
            )