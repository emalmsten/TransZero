import torch

import models
from networks.abstract_network import AbstractNetwork
from models import mlp
import torch.nn as nn
import math
from networks.resnet import DownSample, DownsampleCNN, conv3x3, ResidualBlock, DownSampleTrans


class MuZeroTransformerNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,

        fc_representation_layers,
        support_size,

        transformer_layers,
        transformer_heads,
        transformer_hidden_size,
        max_seq_length,
        positional_embedding_type,  # sinus or learned

        seq_mode,
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
        num_state_values = None # none for continous
    ):
        super().__init__()
        print("MuZeroTransformerNetwork")
        self.action_space_size = action_space_size
        self.support_size = support_size
        self.full_support_size = 2 * support_size + 1
        self.seq_mode = seq_mode
        self.use_proj = use_proj
        self.representation_network_type = representation_network_type
        self.state_size = state_size
        self.cum_reward = cum_reward
        self.downsample = downsample
        self.positional_embedding_type = positional_embedding_type
        self.num_state_values = num_state_values


        def cond_wrap(net):
            return net if self.seq_mode else torch.nn.DataParallel(net)

        # Transformer components for value prediction
        self.transformer_hidden_size = transformer_hidden_size

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
        elif representation_network_type == "mlp":
            self.representation_network = cond_wrap(
                mlp(
                    observation_shape[0]
                    * observation_shape[1]
                    * observation_shape[2]
                    * (stacked_observations + 1)
                    + stacked_observations * observation_shape[1] * observation_shape[2],
                    fc_representation_layers,
                    encoding_size if self.use_proj else transformer_hidden_size,
                    norm_layer=norm_layer,
                )
            )
        elif representation_network_type == "cnn":
            self.representation_network = cond_wrap(
                ConvRepresentationNet(
                    observation_shape[0] * (stacked_observations + 1) + stacked_observations,
                    observation_shape[1],
                    observation_shape[2],
                    encoding_size if self.use_proj else transformer_hidden_size,
                    norm_layer=norm_layer,
                    conv_layers=conv_layers,
                    fc_layers=fc_layers,

                )
            )
        elif representation_network_type == "none":
            flat_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
            self.representation_network = nn.Linear(flat_size, transformer_hidden_size)

        self.action_embedding = nn.Embedding(action_space_size, transformer_hidden_size)
        if self.state_size is not None:
            # linear embedding
            self.state_embedding = nn.Embedding(self.state_values, transformer_hidden_size) if self.num_state_values else nn.Linear(self.state_size[0], transformer_hidden_size)
            #self.state_embedding =

        self.hidden_state_proj = nn.Linear(encoding_size, transformer_hidden_size) # only used if use_proj is True

        if positional_embedding_type == 'learned':
            self.positional_encoding = nn.Embedding(max_seq_length + 1, transformer_hidden_size)
            if self.state_size is not None:
                self.positional_encoding_state_row = nn.Embedding(self.state_size[1], transformer_hidden_size)
                self.positional_encoding_state_col = nn.Embedding(self.state_size[2], transformer_hidden_size)
        elif positional_embedding_type == 'sinus':
            self.register_buffer(
          'positional_encoding',
                self.sinusoidal_positional_embedding(max_seq_length + 1, transformer_hidden_size)
            )
            if self.state_size is not None:
                self.register_buffer(
                    'positional_encoding_state',
                    self.positionalencoding2d(transformer_hidden_size, self.state_size[1], self.state_size[2])
                )

            #self.positional_encoding_state = nn.Embedding(self.state_size[1] * self.state_size[2], transformer_hidden_size)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=transformer_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=transformer_layers,
            norm = nn.LayerNorm(transformer_hidden_size)
        )

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

            self.policy_head = nn.Linear(transformer_hidden_size, self.action_space_size)
            self.value_head = nn.Linear(transformer_hidden_size, self.full_support_size)
            self.reward_head = nn.Linear(transformer_hidden_size, self.full_support_size)


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


    def prediction(self, root_hidden_state, action_sequence=None):

        input_sequence = self.create_input_sequence(root_hidden_state, action_sequence)

        # Pass through the transformer encoder
        #print(input_sequence.size())

        transformer_output = self.transformer_encoder(input_sequence)  # Shape: (B, sequence_length, transformer_hidden_size)

        # Obtain the value prediction from the last token's output
        transformer_output_last = transformer_output[:, -1, :]  # Shape: (B, transformer_hidden_size)

        policy_logits = self.policy_head(transformer_output_last)  # Shape: (B, action_space_size)
        value = self.value_head(transformer_output_last)  # Shape: (B, full_support_size)

        # todo check losses and which values and rewards to include there
        fixed_support_in_self_play = False # todo
        # calculate cumulative reward over sequence
        if action_sequence is not None and self.cum_reward and fixed_support_in_self_play:
            reward = self.reward_head(transformer_output)
            scalars = []
            for i in range(1, reward.size(1)):
                scalars.append(models.support_to_scalar(reward[:, i, :], self.support_size).item())
            # sum all the scalars
            reward = sum(scalars)

        else:
            reward = self.reward_head(transformer_output_last)  # Shape: (B, full_support_size)
            #reward = models.support_to_scalar(reward, self.support_size) # todo

        return policy_logits, value, reward


    def create_causal_mask(self, seq_length):
        return torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()


    def prediction_fast(self, root_hidden_state, action_sequence, action_mask):
        if self.state_size is not None:
            flat_size = self.state_size[1] * self.state_size[2]
            # append false to action mask beginning
            action_mask = torch.cat([torch.zeros(action_mask.size(0), (flat_size-1), dtype=torch.bool, device=action_mask.device), action_mask], dim=1)

        input_sequence = self.create_input_sequence(root_hidden_state, action_sequence) # Shape: (B, sequence_length, transformer_hidden_size)
        causal_mask = self.create_causal_mask(input_sequence.size(1)).to(input_sequence.device) # Shape: (sequence_length, sequence_length)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_sequence, mask=causal_mask, src_key_padding_mask=action_mask)  # Shape: (B, sequence_length, transformer_hidden_size)

        policy_logits = self.policy_head(transformer_output)  # Shape: (B, sequence_length, action_space_size)
        value = self.value_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)
        reward = self.reward_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)

        if self.state_size is not None:
            reward = reward[:, (flat_size - 1):, :]
            value = value[:, (flat_size - 1):, :]
            policy_logits = policy_logits[:, (flat_size - 1):, :]

        return policy_logits, value, reward


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
        elif self.representation_network_type == "mlp":
            return self.representation_mlp(observation)
        elif self.representation_network_type == "cnn":
            return self.representation_res(observation)
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
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
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



    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        #encoded_state_old = self.representation_old(observation)
        # print("encoded_state", encoded_state.shape())
        # print("encoded_state_old", encoded_state_old.shape())
        policy_logits, value, reward = self.prediction(encoded_state)

        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def get_positional_encoding(self, sequence_length, embedded_actions):
        if isinstance(self.positional_encoding, nn.Embedding):
            # Learned positional encoding
            pos_indices = torch.arange(sequence_length, device=embedded_actions.device).unsqueeze(0)  # Shape: (1, sequence_length)
            pos_encoding = self.positional_encoding(pos_indices)  # Shape: (1, sequence_length, transformer_hidden_size)
            #pos_encoding = pos_encoding.expand(embedded_actions.size(0), -1, -1)  # Shape: (B, sequence_length, transformer_hidden_size)
        else:
            # Sinusoidal positional encoding
            pos_encoding = self.positional_encoding[:sequence_length]  # Shape: (sequence_length, transformer_hidden_size)
            pos_encoding = pos_encoding.unsqueeze(0) #.expand(embedded_actions.size(0), -1, -1)  # Shape: (B, sequence_length, transformer_hidden_size)

        return pos_encoding

    def get_positional_encoding_2d(self, H, W, embedded_actions, device):
        if self.positional_embedding_type == 'learned':
            # Learned positional encoding
            row_indices = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).reshape(-1)  # [H*W]
            col_indices = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).reshape(-1)  # [H*W]
            # both_indices = torch.arange(H*W, device=rhs.device) # [H*W, 2]

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



    def create_input_sequence(self, rhs, action_sequence):
        # root hidden state: (B, x)
        # action sequence: (B, y)
        B = rhs.size(0)

        # Embed the action sequence
        if action_sequence is None:
            # Create an empty embedded_actions tensor
            embedded_actions = torch.empty(B, 0, self.transformer_hidden_size, device=rhs.device)
        else:
            # Embed the action sequence
            embedded_actions = self.action_embedding(action_sequence)  # Shape: (B, y, transformer_hidden_size)

        # Total sequence length (including the root hidden state)
        sequence_length_ac = embedded_actions.size(1) # +1 for the root hidden state

        # Get positional encoding
        pos_encoding_ac = self.get_positional_encoding(sequence_length_ac,
                                                    embedded_actions)  # Shape: (B, y+1 transformer_hidden_size)

        embedded_actions = embedded_actions + pos_encoding_ac

        if self.state_size is not None:
            C, H, W = self.state_size
            assert (rhs.size(2)) == H and (rhs.size(3) == W)

            # Get positional encoding for the state
            pos_encoding_state = self.get_positional_encoding_2d(H, W, embedded_actions, rhs.device)

            # flatten out dim 1 and dim 2
            rhs = rhs.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if self.representation_network_type == "none":
                # change dtype to int
                if self.num_state_values:
                    rhs = rhs.int()
                #rhs = rhs.int()
                rhs = self.state_embedding(rhs)
                rhs = rhs.view(B, H * W, self.transformer_hidden_size)

            if self.use_proj:
                rhs = self.hidden_state_proj(rhs)
        else:
            rhs = rhs.unsqueeze(1)  # Shape: (B, 1, transformer_hidden_size)
            pos_encoding_state = torch.zeros(B, 1, self.transformer_hidden_size, device=rhs.device)

        # Concatenate the root hidden state with the positional encoding
        rhs = rhs + pos_encoding_state

        # Project the root hidden state to the transformer hidden size
        if self.use_proj:
            rhs = self.hidden_state_proj(rhs)

        # Construct the input sequence by concatenating the state embedding and action embeddings
        state_action_sequence = torch.cat([rhs, embedded_actions], dim=1)  # Shape: (B, y+1, transformer_hidden_size)

        # Add positional encodings
        return state_action_sequence


    def recurrent_inference_fast(self, root_hidden_state, action_sequence, mask):
        policy_logits, value, reward = self.prediction_fast(root_hidden_state, action_sequence, mask)
        return value, reward, policy_logits


    def recurrent_inference(self, encoded_state, action, root_hidden_state=None, action_sequence=None):
        assert action_sequence is not None, "Transformer needs an action sequence"
        assert root_hidden_state is not None, "Transformer needs a hidden state"

        policy_logits, value, reward = self.prediction(root_hidden_state, action_sequence)

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


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRepresentationNet(nn.Module):
    def __init__(
        self,
        input_channels,
        height,
        width,
        encoding_size,
        conv_layers=None,
        fc_layers=None,
        activation=nn.ELU,
        norm_layer=False
    ):
        """
        Args:
            input_channels (int): Number of input channels
                                  (including stacking of observations if applicable).
            height (int): Height of the input image.
            width (int): Width of the input image.
            encoding_size (int): Desired dimensionality for the final output (e.g., 64, 128).
            conv_layers (list of tuples): Each tuple is (out_channels, kernel_size, stride).
            fc_layers (list of int): Sizes of hidden fully-connected layers after convolution.
            activation (nn.Module): Activation to use between layers (default nn.ELU).
            norm_layer (bool): Whether to apply LayerNorm to the final representation.
        """

        super().__init__()

        if conv_layers is None:
            # Default conv architecture, you can customize these
            conv_layers = [
                # (out_channels, kernel_size, stride)
                (16, 1, 1),  # Output: (batch_size, 16, 3, 3)
                (32, 1, 1),  # Output: (batch_size, 32, 1, 1)
            ]

        if fc_layers is None:
            fc_layers = [64]

        # Build the convolutional "feature extractor"
        conv_modules = []
        current_channels = input_channels
        for (out_channels, kernel_size, stride) in conv_layers:
            conv_modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride))
            conv_modules.append(activation())
            current_channels = out_channels

        self.conv_net = nn.Sequential(*conv_modules)

        # We need to figure out the size of the flattened conv output to build FC layers
        # One way is to do a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            conv_out = self.conv_net(dummy_input)
            self.flattened_size = conv_out.numel()

        # Build the fully-connected layers that project into `encoding_size`
        fc_modules = []
        in_size = self.flattened_size
        for hidden_size in fc_layers:
            fc_modules.append(nn.Linear(in_size, hidden_size))
            fc_modules.append(activation())
            in_size = hidden_size

        # Final layer to produce `encoding_size`
        fc_modules.append(nn.Linear(in_size, encoding_size))

        if norm_layer:
            fc_modules.append(nn.LayerNorm(encoding_size))

        self.fc_net = nn.Sequential(*fc_modules)

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, height, width)
        returns shape: (batch_size, encoding_size)
        """
        conv_features = self.conv_net(x)             # -> (batch_size, channels, H', W')
        flat_features = conv_features.view(x.size(0), -1)  # flatten
        out = self.fc_net(flat_features)             # -> (batch_size, encoding_size)
        return out

