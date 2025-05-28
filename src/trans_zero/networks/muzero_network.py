from trans_zero.networks.fully_connected import MuZeroFullyConnectedNetwork
from trans_zero.networks.resnet import MuZeroResidualNetwork
from trans_zero.networks.transformer import MuZeroTransformerNetwork
from trans_zero.networks.mixed_network import MuZeroMixedNetwork
from trans_zero.networks.double_network import MuZeroDoubleNetwork


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

                res_blocks=config.blocks,
                res_channels=config.channels,
                downsample=config.downsample,

                norm_layer = config.norm_layer,
                use_proj=config.use_proj,
                representation_network_type=config.representation_network_type,

                conv_layers=config.conv_layers_trans,
                fc_layers=config.fc_layers_trans,
                mlp_head_layers=config.mlp_head_layers,
                cum_reward=config.cum_reward,
                state_size=config.state_size,
                stable_transformer=config.stable_transformer,
                config=config, # TODO important, temporary
                vit_params = {
                    "vit_patch_size": config.vit_patch_size,
                    "vit_depth": config.vit_depth,
                    "vit_heads": config.vit_heads,
                    "vit_mlp_dim": config.vit_mlp_dim,
                    "vit_dropout": config.vit_dropout,
                    "use_simple_vit": config.use_simple_vit,
                }
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
            #trans_network.representation_network = trans_network.representation_network

            return MuZeroDoubleNetwork(
                trans_network,
                fully_network,
            )

        else:
            raise NotImplementedError(
                f'The network parameter should be "transformer", "fully_connected" or "resnet". Received: {config.network}'
            )