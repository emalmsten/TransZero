import torch
try:
    from vit_pytorch import ViT, SimpleViT
except ImportError:
    print("VitPytorch not installed, install to use ViT representation network.")

from trans_zero.utils import models
from trans_zero.networks.abstract_network import AbstractNetwork
from trans_zero.utils.models import mlp
import torch.nn as nn
import math
from trans_zero.networks.resnet import DownSample, DownsampleCNN, conv3x3, ResidualBlock, DownSampleTrans
from trans_zero.networks.stable_transformer import StableTransformerXL


class MuZeroTransformerNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,

        fc_representation_layers,
        support_size,

        max_seq_length,
        positional_embedding_type,  # sinus or learned

        seq_mode,

        transformer_params,

        norm_layer = True,
        use_proj = False,

        conv_layers = None,
        fc_layers = None,

        res_blocks = 3,
        res_channels = 8,
        downsample = None,

        representation_network_type = "res",
        mlp_head_layers = None,
        cum_reward = False,
        state_size = None,
        num_state_values = None, # none for continous,
        stable_transformer = True,

        config = None,
        vit_params = None, # vit_patch_size, vit_depth, vit_heads, vit_mlp_dim
    ):
        super().__init__()
        self.config = config


        print("MuZeroTransformerNetwork")
        self.action_space_size = action_space_size
        self.support_size = support_size
        self.full_support_size = 2 * support_size + 1 if support_size > 1 else 1
        self.seq_mode = seq_mode
        self.use_proj = use_proj
        self.representation_network_type = representation_network_type
        self.state_size = state_size
        self.cum_reward = cum_reward
        self.downsample = downsample
        self.positional_embedding_type = positional_embedding_type
        self.num_state_values = num_state_values
        self.stable_transformer = stable_transformer


        def cond_wrap(net):
            return net if self.seq_mode else torch.nn.DataParallel(net)

        # Transformer components for value prediction
        self.transformer_hidden_size = transformer_params["transformer_hidden_size"]

        if representation_network_type == "res":
            self.representation_network = cond_wrap(
                RepresentationNetwork(
                    observation_shape,
                    stacked_observations,
                    num_blocks=res_blocks,
                    num_channels=res_channels,
                    downsample=downsample,
                    fc_layers=fc_layers,
                    encoding_size=self.transformer_hidden_size,
                    norm_layer=norm_layer,
                    state_size=self.state_size
                )
            )

        elif representation_network_type == "res_pool":
            self.representation_network = cond_wrap(
                RepresentationNetworkPool(
                    observation_shape,
                    stacked_observations,
                    num_blocks=res_blocks,
                    num_channels=64,
                    downsample=downsample,
                    fc_layers=fc_layers,
                    encoding_size=self.transformer_hidden_size,
                    norm_layer=norm_layer,
                    state_size=self.state_size
                )
            )

        elif representation_network_type == "mlp":
            self.representation_network = cond_wrap(
                mlp(
                    observation_shape[0]
                    * observation_shape[1]
                    * observation_shape[2]
                    * (stacked_observations + 1)
                    + stacked_observations * observation_shape[1] * observation_shape[2],
                    fc_representation_layers,
                    encoding_size if self.use_proj else self.transformer_hidden_size,
                    norm_layer=norm_layer,
                )
            )
        elif representation_network_type == "cnn_mlp":
            self.cnn = ConvRepresentationNet(
                input_channels=6,
                conv_layers=self.config.conv_layers_trans,
            )

            self.representation_network = cond_wrap(
                mlp(
                    conv_layers[-1][0]
                    * observation_shape[1]
                    * observation_shape[2]
                    * (stacked_observations + 1)
                    + stacked_observations * observation_shape[1] * observation_shape[2],
                    fc_representation_layers,
                    encoding_size if self.use_proj else self.transformer_hidden_size,
                    norm_layer=norm_layer,
                )
            )
        elif representation_network_type == "cnn_pool":
            self.representation_network = cond_wrap(MiniGridCNN())

        elif representation_network_type == "cnn_pool2":
            self.representation_network = cond_wrap(
                CNNPool(
                    input_channels=6,
                    transformer_hidden_size=self.transformer_hidden_size,
                    conv_configs=self.config.conv_pool_config,
                    fc_layers=fc_layers,
                )
            )

        elif representation_network_type == "ViT":
            assert vit_params is not None, "vit_params must be provided for ViT representation network"
            self.representation_network = cond_wrap(
                RepViT(
                    in_channels = observation_shape[0],
                    size = (observation_shape[1], observation_shape[2]),
                    transformer_hidden_size= self.transformer_hidden_size,
                    vit_params = vit_params,
                )
            )

        elif representation_network_type == "cls":
            self.register_buffer(
                'positional_encoding_state',
                self.positionalencoding2d(self.transformer_hidden_size, self.state_size[1], self.state_size[2])
            )

            self.representation_network = cond_wrap(
                RepWithCLS(6, self.positional_encoding_state)
            )

        elif representation_network_type == "cls_adv":


            self.representation_network = cond_wrap(
                AdvancedRepWithCLS(
                    observation_shape,
                    stacked_observations,
                    num_blocks=res_blocks,
                    num_channels=self.transformer_hidden_size,
                    downsample=downsample,
                )
            )


        elif representation_network_type == "cnn":
            self.representation_network = cond_wrap(
                ConvRepresentationNet(
                    observation_shape[0] * (stacked_observations + 1) + stacked_observations,
                    observation_shape[1],
                    observation_shape[2],
                    encoding_size if self.use_proj else self.transformer_hidden_size,
                    norm_layer=norm_layer,
                    conv_layers=conv_layers,
                    fc_layers=fc_layers,

                )
            )
        elif representation_network_type == "none":
            flat_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
            self.representation_network = nn.Linear(flat_size, self.transformer_hidden_size)

        self.action_embedding = nn.Embedding(action_space_size, self.transformer_hidden_size)

        # only used if rep net is none
        if self.state_size is not None:
            # linear embedding
            self.state_embedding = nn.Embedding(self.state_values, self.transformer_hidden_size) if self.num_state_values else nn.Linear(self.state_size[0], transformer_hidden_size)
            #self.state_embedding =

        self.hidden_state_proj = nn.Linear(encoding_size, self.transformer_hidden_size) # only used if use_proj is True

        if positional_embedding_type == 'learned':
            self.positional_encoding = nn.Embedding(max_seq_length + 1, self.transformer_hidden_size)
            if self.state_size is not None:
                self.positional_encoding_state_row = nn.Embedding(self.state_size[1], self.transformer_hidden_size)
                self.positional_encoding_state_col = nn.Embedding(self.state_size[2], self.transformer_hidden_size)
        elif positional_embedding_type == 'sinus':
            self.register_buffer(
          'positional_encoding',
                self.sinusoidal_positional_embedding(max_seq_length + 1, self.transformer_hidden_size)
            )
            if self.state_size is not None:
                self.register_buffer(
                    'positional_encoding_state',
                    self.positionalencoding2d(self.transformer_hidden_size, self.state_size[1], self.state_size[2])
                )

            #self.positional_encoding_state = nn.Embedding(self.state_size[1] * self.state_size[2], transformer_hidden_size)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_params["transformer_hidden_size"],
            nhead=transformer_params["transformer_heads"],
            dim_feedforward=transformer_params["transformer_mlp_dim"],
            dropout=transformer_params["transformer_dropout"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=transformer_params["transformer_layers"],
            norm = nn.LayerNorm(self.transformer_hidden_size)
        )

        if self.stable_transformer:
            self.transformer_encoder = StableTransformerXL(
                d_input=transformer_params["transformer_hidden_size"],
                n_layers=transformer_params["transformer_layers"],
                n_heads=transformer_params["transformer_heads"],
                d_head_inner=self.transformer_hidden_size // transformer_params["transformer_heads"],  # Adjust if needed
                d_ff_inner=4 * self.transformer_hidden_size,  # Common practice
                dropout=0.1,
                dropouta=0.0,
                mem_len=100,  # Adjust memory length as per requirement
            )
            self.memory = self.transformer_encoder.init_memory(1) # batch size

        if mlp_head_layers is not None:
            self.value_head = mlp(
                self.transformer_hidden_size,
                mlp_head_layers,
                self.full_support_size
            )
            self.policy_head = mlp(
                self.transformer_hidden_size,
                mlp_head_layers,
                action_space_size,
            )
            self.reward_head = mlp(
                self.transformer_hidden_size,
                mlp_head_layers,
                self.full_support_size,
            )
        else:
            self.policy_head = nn.Linear(self.transformer_hidden_size, self.action_space_size)
            self.value_head = nn.Linear(self.transformer_hidden_size, self.full_support_size)
            self.reward_head = nn.Linear(self.transformer_hidden_size, self.full_support_size)


    def sinusoidal_positional_embedding(self, length, d_model, n=10000.0):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model}).")

        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(n) / d_model)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def get_cumulative_reward(self, transformer_output):
        reward = self.reward_head(transformer_output)
        scalars = []
        for i in range(1, reward.size(1)):
            scalars.append(models.support_to_scalar(reward[:, i, :], self.support_size).item())
        # sum all the scalars
        return sum(scalars)


    def prediction(self, latent_root_state, action_sequence=None, custom_pos_indices=None, custom_causal_mask=None,
                   return_n_last_predictions=1):

        if action_sequence is None and self.config.use_s0_for_pred:
            return self.policy_head(latent_root_state), self.value_head(latent_root_state), None

        input_sequence = self.create_input_sequence(latent_root_state, action_sequence, custom_pos_indices=custom_pos_indices)

        # Pass through the transformer encoder
        causal_mask = None

        if custom_causal_mask is not None:
            causal_mask = custom_causal_mask
        elif self.config.use_forward_causal_mask:
            causal_mask = self.create_causal_mask(input_sequence.size(1)).to(
                input_sequence.device)  # Shape: (sequence_length, sequence_length)

        if self.stable_transformer:
            input_sequence = input_sequence.transpose(0, 1)

            result = self.transformer_encoder(input_sequence)
            transformer_output = result["logits"]
            transformer_output = transformer_output.transpose(0, 1)
        else:
            is_causal = causal_mask is not None and custom_causal_mask is None
            transformer_output = self.transformer_encoder(input_sequence, mask=causal_mask, is_causal=is_causal)  # Shape: (B, sequence_length, transformer_hidden_size)

        # Shape: (B, sequence_length, transformer_hidden_size)
        if self.config.use_s0_for_pred:
            # Use the latent root state as the transformer output
            transformer_output[:, 0, :] = latent_root_state

        # Obtain the value prediction from the last token's output
        if return_n_last_predictions == 1:
            transformer_output_for_prediction = transformer_output[:, -1, :]  # Shape: (B, transformer_hidden_size)
        else:
            transformer_output_for_prediction = transformer_output[:, -return_n_last_predictions:, :]
            transformer_output_for_prediction = transformer_output_for_prediction.squeeze(0)


        policy_logits = self.policy_head(transformer_output_for_prediction)  # Shape: (B, action_space_size)
        value = self.value_head(transformer_output_for_prediction)  # Shape: (B, full_support_size)

        # todo check losses and which values and rewards to include there
        fixed_support_in_self_play = False # todo
        # calculate cumulative reward over sequence
        if action_sequence is not None and self.cum_reward and fixed_support_in_self_play:
            reward = self.get_cumulative_reward(transformer_output_for_prediction)
        else:
            reward = self.reward_head(transformer_output_for_prediction)  # Shape: (B, full_support_size)
            #reward = models.support_to_scalar(reward, self.support_size) # todo

        return policy_logits, value, reward


    def create_causal_mask(self, seq_length):
        return torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)


    def stable_transformer_forward(self, input_sequence, mask):
        input_sequence = input_sequence.transpose(0, 1)

        result = self.transformer_encoder(input_sequence, src_key_padding_mask=mask)
        transformer_output = result["logits"]
        return transformer_output.transpose(0, 1)


    def prediction_fast(self, latent_root_state, action_sequence, action_mask, use_causal_mask=True):
        #action_mask = None
        if self.state_size is not None:
            flat_size = self.state_size[1] * self.state_size[2]
            # append false to action mask beginning
            action_mask = torch.cat([torch.zeros(action_mask.size(0), (flat_size-1), dtype=torch.bool, device=action_mask.device), action_mask], dim=1)

        #action_mask = action_mask.float().masked_fill(action_mask, float('-inf'))

        input_sequence = self.create_input_sequence(latent_root_state, action_sequence) # Shape: (B, sequence_length, transformer_hidden_size)
        if use_causal_mask:
            causal_mask = self.create_causal_mask(input_sequence.size(1)).to(input_sequence.device) # Shape: (sequence_length, sequence_length)
        else:
            causal_mask = None

        # Pass through the transformer encoder
        if self.stable_transformer:
            transformer_output = self.stable_transformer_forward(input_sequence, causal_mask)
        else:
            transformer_output = self.transformer_encoder(input_sequence, mask=causal_mask)#, src_key_padding_mask=action_mask)  # Shape: (B, sequence_length, transformer_hidden_size)

        policy_logits = self.policy_head(transformer_output)  # Shape: (B, sequence_length, action_space_size)
        value = self.value_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)
        reward = self.reward_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)

        # only return the predicted action tokens
        if self.state_size is not None:
            reward = reward[:, (flat_size - 1):, :]
            value = value[:, (flat_size - 1):, :]
            policy_logits = policy_logits[:, (flat_size - 1):, :]

        return policy_logits, value, reward, transformer_output


    def random_prediction(self, device):
        policy_logits = torch.rand((1, self.action_space_size), device=device, requires_grad=True)
        value = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        reward = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        return policy_logits, value, reward

    def representation(self, observation):
        if self.representation_network_type == "res":
            if self.state_size is not None:
                return self.representation_3d(observation)
            else:
                return self.representation_res(observation)
        elif self.representation_network_type == "res_pool":
            return self.representation_res(observation)
        elif self.representation_network_type == "mlp":
            return self.representation_mlp(observation)
        elif self.representation_network_type in ["cnn", "cnn_pool", "cnn_pool2"]:
            return self.representation_res(observation)
        elif self.representation_network_type == "cnn_mlp":
            enc_obs = self.cnn(observation)
            return self.representation_mlp(enc_obs)



        elif self.representation_network_type == "cls" or self.representation_network_type == "cls_adv" or self.representation_network_type == "ViT":
            return self.representation_network(observation)

        elif self.representation_network_type == "none":

            # flatten observation
            if self.state_size is not None:
                return observation


            observation = observation.view(observation.size(0), -1)
            observation = self.representation_network(observation)
            # pad observation with -1 until size (B, 64)
            #observation = F.pad(observation, (0, 64 - observation.size(1)), "constant", -1)
            return observation
        else:
            raise NotImplementedError(f"representation_network_type {self.representation_network_type} not implemented")


    def representation_3d(self, observation):
        # observation = (
        #     torch.tensor(observation)
        #     .float()
        #     .unsqueeze(0)
        #     .to(observation.device)
        # )

        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
        return encoded_state_normalized


    def representation_res(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = ( encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized


    def representation_mlp(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def representation_cnn(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized



    def pll_initial_inference(self, encoded_state, state_reward, pll_args):
        all_policy_logits, values, rewards = self.prediction(
            encoded_state,
            **pll_args
        )

        # reward at root equal to 0 for consistency
        rewards[0] = state_reward

        return values, rewards, all_policy_logits, encoded_state


    def initial_inference(self, observation, just_state=False, pll_args=None):
        # only in the parallel version are the optional arguments used

        encoded_state = self.representation(observation)

        # reward at root equal to 0 for consistency
        state_reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        if just_state:
            return None, state_reward, None, encoded_state

        if pll_args is not None:
            return self.pll_initial_inference(encoded_state, state_reward, pll_args)

        policy_logits, value, _ = self.prediction(encoded_state)

        return (
            value,
            state_reward,
            policy_logits,
            encoded_state,
        )

    def get_positional_encoding(self, sequence_length, embedded_actions, pos_indices=None):
        if isinstance(self.positional_encoding, nn.Embedding):
            if pos_indices is None:
                pos_indices = torch.arange(sequence_length, device=embedded_actions.device).unsqueeze(0)
            else:
                # Ensure the provided indices have the correct dimensions (e.g., unsqueeze if they are 1D)
                if pos_indices.dim() == 1:
                    pos_indices = pos_indices.unsqueeze(0)
            pos_encoding = self.positional_encoding(pos_indices)
        else:
            if pos_indices is None:
                # Default to the contiguous range if no pos_indices provided
                pos_encoding = self.positional_encoding[:sequence_length]
            else:
                # Use the provided indices to fetch the corresponding rows from the sinusoidal encoding.
                # Here, we assume pos_indices is a tensor of indices into the precomputed positional encodings.
                pos_encoding = self.positional_encoding[pos_indices.squeeze(0)]
            pos_encoding = pos_encoding.unsqueeze(0)  # Make sure dimensions match (B, sequence_length, transformer_hidden_size)

        return pos_encoding

    def get_positional_encoding_2d(self, H, W, embedded_actions, device):
        if self.positional_embedding_type == 'learned':
            # Learned positional encoding
            row_indices = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).reshape(-1)  # [H*W]
            col_indices = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).reshape(-1)  # [H*W]
            # both_indices = torch.arange(H*W, device=lrs.device) # [H*W, 2]

            # Get embeddings and sum them
            pe = self.positional_encoding_state_row(row_indices) + self.positional_encoding_state_col(
                col_indices)  # [H*W, transformer_hidden_size]
            # Expand to match the batch dimension:
            pe = pe.unsqueeze(0)  # [1, H*W, transformer_hidden_size]

        elif self.positional_embedding_type == 'sinus':
            pe = self.positionalencoding2d(self.transformer_hidden_size, H, W).to(device)  # [transformer_hidden_size, H, W]
            pe = pe.view(1, -1, self.transformer_hidden_size)
        else:
            raise ValueError(f"Unknown positional encoding type: {self.positional_encoding_type}")

        return pe



    def create_input_sequence(self, lrs, action_sequence, custom_pos_indices=None):
        # root hidden state: (B, x)
        # action sequence: (B, y)
        B = lrs.size(0)

        # Embed the action sequence
        if action_sequence is None:
            # Create an empty embedded_actions tensor
            embedded_actions = torch.empty(B, 0, self.transformer_hidden_size, device=lrs.device)
        else:
            # Embed the action sequence
            embedded_actions = self.action_embedding(action_sequence)  # Shape: (B, y, transformer_hidden_size)

        # Total sequence length (including the root hidden state)
        sequence_length_ac = embedded_actions.size(1) # +1 for the root hidden state

        # Get positional encoding
        pos_encoding_ac = self.get_positional_encoding(sequence_length_ac, embedded_actions, pos_indices=custom_pos_indices)  # Shape: (B, y+1 transformer_hidden_size)
        embedded_actions = embedded_actions + pos_encoding_ac


        if self.state_size is not None:
            C, H, W = self.state_size
            assert (lrs.size(2)) == H and (lrs.size(3) == W)

            # Get positional encoding for the states
            pos_encoding_state = self.get_positional_encoding_2d(H, W, embedded_actions, lrs.device)

            # flatten out dim 1 and dim 2
            lrs = lrs.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # if rep net hasn't made an encoding
            if self.representation_network_type == "none":
                # if state values are discrete, embed them
                if self.num_state_values:
                    lrs = lrs.int()
                lrs = self.state_embedding(lrs)
                lrs = lrs.view(B, H * W, self.transformer_hidden_size)

            if self.use_proj:
                lrs = self.hidden_state_proj(lrs)
        else:
            lrs = lrs.unsqueeze(1)  # Shape: (B, 1, transformer_hidden_size)
            pos_encoding_state = torch.zeros(B, 1, self.transformer_hidden_size, device=lrs.device)

        # Project the root hidden state to the transformer hidden size
        if self.use_proj:
            lrs = self.hidden_state_proj(lrs)

        # Concatenate the root hidden state with the positional encoding
        lrs = lrs + pos_encoding_state

        # Construct the input sequence by concatenating the state embedding and action embeddings
        state_action_sequence = torch.cat([lrs, embedded_actions], dim=1)  # Shape: (B, y+1, transformer_hidden_size)

        # Add positional encodings
        return state_action_sequence


    def recurrent_inference_fast(self, latent_root_state, action_sequence, mask, use_causal_mask=True):
        policy_logits, value, reward, transformer_output = self.prediction_fast(latent_root_state, action_sequence, mask, use_causal_mask=use_causal_mask)
        return value, reward, policy_logits, transformer_output


    def recurrent_inference(self, encoded_state, action, latent_root_state=None, action_sequence=None,
                            custom_pos_indices=None, custom_causal_mask=None, return_n_last_predictions=1):
        assert action_sequence is not None, "Transformer needs an action sequence"
        assert latent_root_state is not None, "Transformer needs a hidden state"

        policy_logits, value, reward = self.prediction(latent_root_state, action_sequence,
                                                       custom_pos_indices=custom_pos_indices,
                                                       custom_causal_mask =custom_causal_mask,
                                                       return_n_last_predictions=return_n_last_predictions)

        return value, reward, policy_logits, None # next encoded state



class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
        fc_layers = None,
        encoding_size = 32,
        activation=nn.ELU,
        norm_layer=True,
        state_size = None

    ):
        super().__init__()
        self.state_size = state_size
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet_s":
                self.downsample_net = DownSampleTrans(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')

        else:
            if observation_shape[1] == 1 or observation_shape[2] == 1:
                self.conv_type = "1x1"
            else:
                self.conv_type = "3x3"
            # self.conv_type = "1x1" # todo overwrite

            self.conv = conv3x3(
                observation_shape[0] * (stacked_observations + 1) + stacked_observations,
                num_channels, conv_type = self.conv_type
            )

            self.bn = torch.nn.BatchNorm2d(num_channels)


        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, observation_shape[0], observation_shape[1], observation_shape[2])
            conv_out = self.conv(dummy_input) if not self.downsample else self.downsample_net(dummy_input)
            self.flattened_size = conv_out.numel()
            print("flattened_size", self.flattened_size)

        if fc_layers is None:
            fc_layers = [128, 64]

        # Build the fully-connected layers that project into `encoding_size`
        fc_modules = []
        in_size = self.flattened_size
        for hidden_size in fc_layers:
            fc_modules.append(nn.Linear(in_size, hidden_size))
            fc_modules.append(activation())
            in_size = hidden_size

        # Final layer to produce `encoding_size`
        fc_modules.append(nn.Linear(in_size, encoding_size))
        #fc_modules.append(activation())

        if norm_layer:
            fc_modules.append(nn.LayerNorm(encoding_size))

        self.fc_net = nn.Sequential(*fc_modules)



    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        if self.state_size is None: # TODO
            x = x.view(x.size(0), -1)  # flatten
            x = self.fc_net(x)

        return x



class RepresentationNetworkPool(nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
        fc_layers=None,
        encoding_size=32,
        activation=nn.ELU,
        norm_layer=True,
        state_size=None
    ):
        super().__init__()
        self.state_size = state_size
        self.downsample = downsample

        in_ch = observation_shape[0] * (stacked_observations + 1) + stacked_observations
        if observation_shape[1] == 1 or observation_shape[2] == 1:
            conv_type = "1x1"
        else:
            conv_type = "3x3"
        self.front = nn.Sequential(
            conv3x3(in_ch, num_channels, conv_type=conv_type),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # residual blocks
        self.resblocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        # global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # build FC net
        if fc_layers is None:
            fc_layers = [128, 64]

        layers = []
        in_size = num_channels  # after pool we have [B, num_channels, 1, 1]
        for h in fc_layers:
            layers += [nn.Linear(in_size, h), activation()]
            in_size = h
        layers.append(nn.Linear(in_size, encoding_size))
        if norm_layer:
            layers.append(nn.LayerNorm(encoding_size))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.front(x)
        for block in self.resblocks:
            x = block(x)
        if self.state_size is None:
            x = self.global_pool(x)           # [B, C, 1, 1]
            x = x.view(x.size(0), -1)         # [B, C]
            x = self.fc(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F



class MiniGridCNN(nn.Module):
    def __init__(self, out_channels, fc_layers):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(6, 16, kernel_size=2),  # (6,3,3) -> (16,2,2)
        #     nn.ReLU(),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # → (32, 3, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # → (32, 1, 1)
        )

        self.pool = nn.AdaptiveMaxPool2d((1, 1))  # (16,2,2) -> (16,1,1)
        self.flatten = nn.Flatten()               # -> (16)
        #self.fc = nn.Linear(32, 32)               # -> (32)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        #x = self.fc(x)
        return x



import torch
import torch.nn as nn

class CNNPool(nn.Module):
    """
    A flexible CNN for arbitrary-size grid inputs that uses optional per-layer max pooling
    and ends with a 1×1 adaptive max pool before optional FC layers.

    This module accepts any spatial dimensions (height, width) at runtime, thanks to the
    final AdaptiveMaxPool2d((1,1)).

    Args:
        input_channels (int): Number of channels in the input tensor.
        conv_configs (list of dict, optional): Specs for each conv block. Each dict may include:
            - out_channels (int): number of filters.
            - kernel_size (int or tuple): conv kernel size (default=3).
            - stride (int or tuple): conv stride (default=1).
            - padding (int or tuple): conv padding (default=0).
            - pool_kernel (int or tuple, optional): if set, applies MaxPool2d after ReLU.
        fc_layers (list of int, optional): Sizes for fully connected layers after flattening.
    """
    def __init__(self,
                 input_channels=6,
                 transformer_hidden_size=32,
                 conv_configs=None,
                 fc_layers=None,
                 activation=nn.ELU):
        super().__init__()

        if conv_configs is None:
            conv_configs = [
                {'out_channels': 32, 'kernel_size': 3, 'padding': 1}
            ]

        layers = []
        in_ch = input_channels
        for cfg in conv_configs:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    cfg['out_channels'],
                    kernel_size=cfg.get('kernel_size', 3),
                    stride=cfg.get('stride', 1),
                    padding=cfg.get('padding', 0)
                )
            )
            layers.append(activation())
            if 'pool_kernel' in cfg and cfg['pool_kernel']:
                layers.append(nn.MaxPool2d(kernel_size=cfg['pool_kernel']))
            in_ch = cfg['out_channels']

        self.conv = nn.Sequential(*layers)
        # final adaptive max pool to 1×1 ensures spatial agnosticism
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))

        # optional fully connected head
        if fc_layers is None:
            fc_layers = []
        fc_modules = []
        in_size = in_ch  # after conv+pool, shape is [B, in_ch, 1, 1]
        for h in fc_layers:
            fc_modules.append(nn.Linear(in_size, h))
            fc_modules.append(activation())
            in_size = h

        # final proj to transformer size
        fc_modules.append(nn.Linear(in_size, transformer_hidden_size))

        # *** use the fc_modules list here! ***
        self.fc = nn.Sequential(*fc_modules)


    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channels, height, width)
        Returns:
            Tensor: shape (batch, fc_sizes[-1] or in_channels) after adaptive pooling and optional FC
        """
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        return x

# Example usage:
# ----------------------------------------------------------------
# Any input spatial size works, e.g., 5×5, 8×8, 10×10, etc.
# model = CNNFeatureExtractor(
#     input_channels=6,
#     conv_configs=[
#         {'out_channels': 16, 'kernel_size': 3, 'padding': 1, 'pool_kernel': 2},
#         {'out_channels': 32, 'kernel_size': 3, 'padding': 1},
#         {'out_channels': 64, 'kernel_size': 3, 'padding': 1},
#     ],
#     fc_sizes=[128, 64]
# )
# print(model)



class ConvRepresentationNet(nn.Module):
    def __init__(self, input_channels, conv_layers, activation=nn.ELU):
        super().__init__()
        layers = []
        in_channels = input_channels

        for out_channels, kernel_size, stride in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2))
            layers.append(activation())
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)



    def forward(self, x):
        """
        x shape: (batch_size, input_channels, height, width)
        returns shape: (batch_size, encoding_size)
        """
        return self.encoder(x)

import torch
import torch.nn as nn







class RepViT(nn.Module):
    def __init__(self, in_channels, size, transformer_hidden_size, vit_params):
        super().__init__()
        # stem & residuals as before, ending in (B, 32, 3, 3)…
        if vit_params["use_simple_vit"]:
            print("Using SimpleViT for representation")
            self.model = SimpleViT(
                image_size=size,  # Grid is 3x3
                patch_size=vit_params["vit_patch_size"],  # Use 1x1 patches to preserve all 9 positions
                num_classes=transformer_hidden_size,  # We want a single output token
                dim=transformer_hidden_size,  # Hidden size
                depth=vit_params["vit_depth"],  # Number of transformer layers (can be tuned)
                heads=vit_params["vit_heads"],  # Number of attention heads
                mlp_dim=vit_params["vit_mlp_dim"],  # Dimension of the MLP inside each transformer block
                #dropout=vit_params["vit_dropout"],  # Dropout rate
                channels=in_channels,  # Number of input channels per grid cell
            )
        else:
            self.model = ViT(
                image_size=size,  # Grid is 3x3
                patch_size=vit_params["vit_patch_size"],  # Use 1x1 patches to preserve all 9 positions
                num_classes=1,  # We want a single output token
                dim=transformer_hidden_size,  # Hidden size
                depth=vit_params["vit_depth"],  # Number of transformer layers (can be tuned)
                heads=vit_params["vit_heads"],  # Number of attention heads
                mlp_dim=vit_params["vit_mlp_dim"],  # Dimension of the MLP inside each transformer block
                dropout=vit_params["vit_dropout"],  # Dropout rate
                channels=in_channels,  # Number of input channels per grid cell
                pool='cls'  # Use CLS token for final output
            )

            # To get the CLS token instead of classification:
            self.model.mlp_head = torch.nn.Identity()

    def forward(self, x):
        out = self.model(x)  # Output shape: (batch_size, transformer_hidden_size)
        return out



class RepWithCLS(nn.Module):
    def __init__(self, in_channels, pos_enc):
        super().__init__()
        # stem & residuals as before, ending in (B, 32, 3, 3)…
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
        )
        self.pos_enc = pos_enc
        # one learned CLS token per batch (we'll expand it)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
        # a tiny transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)                    # → (B, 32, 3, 3)
        tokens = x.flatten(2).transpose(1, 2)  # → (B,  9, 32)

        pe = self.positionalencoding2d(self.transformer_hidden_size, 3, 3).to(x.device)  # [transformer_hidden_size, H, W]
        pe = pe.view(-1, -1, self.transformer_hidden_size)
        tokens = tokens + pe  # add positional encoding to tokens

        # prepend CLS:
        cls = self.cls_token.expand(B, -1, -1)  # → (B, 1, 32)
        seq = torch.cat([cls, tokens], dim=1)   # → (B, 10, 32)
        # Transformer wants (seq_len, batch, dim):
        seq = seq.transpose(0, 1)               # → (10, B, 32)
        out = self.encoder(seq)                 # → (10, B, 32)
        cls_out = out[0]                        # → (B, 32)
        return cls_out                       # single state token


class AdvancedRepWithCLS(nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample=None,
        nheads=4,
        nlayers=1,
    ):
        super().__init__()
        # --- build the conv / downsample stem exactly as in RepresentationNetwork ---
        self.downsample = downsample
        in_ch = observation_shape[0] * (stacked_observations + 1) + stacked_observations

        if self.downsample:
            if self.downsample == "resnet_s":
                self.conv_stem = DownSampleTrans(in_ch, num_channels)
            elif self.downsample == "CNN":
                self.conv_stem = DownsampleCNN(
                    in_ch,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample must be "resnet_s" or "CNN"')
        else:
            # regular conv + BN + ReLU
            conv_type = "1x1" if (observation_shape[1] == 1 or observation_shape[2] == 1) else "3x3"
            self.conv_stem = conv3x3(in_ch, num_channels, conv_type=conv_type)
            self.bn = nn.BatchNorm2d(num_channels)

        # residual blocks
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # --- now the CLS token + transformer encoder ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=nheads, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        # conv / downsample stem
        if self.downsample:
            x = self.conv_stem(x)
        else:
            x = self.conv_stem(x)
            x = self.bn(x)
            x = F.relu(x)

        # residual blocks
        for block in self.resblocks:
            x = block(x)

        # make tokens: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # → (B, H*W, C)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # → (B, 1, C)
        seq = torch.cat([cls, tokens], dim=1)   # → (B, 1 + H*W, C)

        # transformer wants (seq_len, batch, dim)
        seq = seq.transpose(0, 1)               # → (1 + H*W, B, C)
        out = self.encoder(seq)                 # → (1 + H*W, B, C)

        # take CLS output
        cls_out = out[0]                        # → (B, C)
        return cls_out

#
# class ConvRepresentationNet(nn.Module):
#     def __init__(
#         self,
#         input_channels,
#         height,
#         width,
#         encoding_size,
#         conv_layers=None,
#         fc_layers=None,
#         activation=nn.ELU,
#         norm_layer=False
#     ):
#         """
#         Args:
#             input_channels (int): Number of input channels
#                                   (including stacking of observations if applicable).
#             height (int): Height of the input image.
#             width (int): Width of the input image.
#             encoding_size (int): Desired dimensionality for the final output (e.g., 64, 128).
#             conv_layers (list of tuples): Each tuple is (out_channels, kernel_size, stride).
#             fc_layers (list of int): Sizes of hidden fully-connected layers after convolution.
#             activation (nn.Module): Activation to use between layers (default nn.ELU).
#             norm_layer (bool): Whether to apply LayerNorm to the final representation.
#         """
#
#         super().__init__()
#
#         if conv_layers is None:
#             # Default conv architecture, you can customize these
#             conv_layers = [
#                 # (out_channels, kernel_size, stride)
#                 (16, 1, 1),  # Output: (batch_size, 16, 3, 3)
#                 (32, 1, 1),  # Output: (batch_size, 32, 1, 1)
#             ]
#
#         if fc_layers is None:
#             fc_layers = [64]
#
#         # Build the convolutional "feature extractor"
#         conv_modules = []
#         current_channels = input_channels
#         for (out_channels, kernel_size, stride) in conv_layers:
#             conv_modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride))
#             conv_modules.append(activation())
#             current_channels = out_channels
#
#         self.conv_net = nn.Sequential(*conv_modules)
#
#         # We need to figure out the size of the flattened conv output to build FC layers
#         # One way is to do a dummy forward pass
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, input_channels, height, width)
#             conv_out = self.conv_net(dummy_input)
#             self.flattened_size = conv_out.numel()
#
#         # Build the fully-connected layers that project into `encoding_size`
#         fc_modules = []
#         in_size = self.flattened_size
#         for hidden_size in fc_layers:
#             fc_modules.append(nn.Linear(in_size, hidden_size))
#             fc_modules.append(activation())
#             in_size = hidden_size
#
#         # Final layer to produce `encoding_size`
#         fc_modules.append(nn.Linear(in_size, encoding_size))
#
#         if norm_layer:
#             fc_modules.append(nn.LayerNorm(encoding_size))
#
#         self.fc_net = nn.Sequential(*fc_modules)