import torch
from networks.abstract_network import AbstractNetwork
from models import mlp
import torch.nn as nn
import math
from networks.resnet import DownSample, DownsampleCNN, conv3x3, ResidualBlock

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

        representation_network_type = "res"
    ):
        super().__init__()
        print("MuZeroTransformerNetwork")
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.seq_mode = seq_mode
        self.use_proj = use_proj
        self.representation_network_type = representation_network_type

        def cond_wrap(net):
            return net if self.seq_mode else torch.nn.DataParallel(net)

        # Transformer components for value prediction
        self.transformer_hidden_size = transformer_hidden_size

        if representation_network_type == "res":
            self.representation_network = cond_wrap(
                RepresentationNetwork(
                    observation_shape,
                    stacked_observations,
                    3,
                    3,
                    None,
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
            # not implemented
            raise NotImplementedError("representation_network_type 'cnn' not implemented")

        self.action_embedding = nn.Embedding(action_space_size, transformer_hidden_size)
        self.hidden_state_proj = nn.Linear(encoding_size, transformer_hidden_size) # only used if use_proj is True

        if positional_embedding_type == 'learned':
            self.positional_encoding = nn.Embedding(max_seq_length + 1, transformer_hidden_size)
        elif positional_embedding_type == 'sinus':
            self.register_buffer(
          'positional_encoding',
                self.sinusoidal_positional_embedding(max_seq_length + 1, transformer_hidden_size)
            )

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


    def prediction(self, root_hidden_state, action_sequence=None):
        input_sequence = self.create_input_sequence(root_hidden_state, action_sequence)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_sequence)  # Shape: (B, sequence_length, transformer_hidden_size)

        # Obtain the value prediction from the last token's output
        transformer_output_last = transformer_output[:, -1, :]  # Shape: (B, transformer_hidden_size)

        policy_logits = self.policy_head(transformer_output_last)  # Shape: (B, action_space_size)
        value = self.value_head(transformer_output_last)  # Shape: (B, full_support_size)

        # calculate cumulative reward over sequence # todo test
        if False:
            reward = self.reward_head(transformer_output[:, 1:, :]).sum(dim=1)
        else:
            reward = self.reward_head(transformer_output_last)  # Shape: (B, full_support_size)

        return policy_logits, value, reward


    def create_causal_mask(self, seq_length):
        return torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()


    def prediction_fast(self, root_hidden_state, action_sequence, action_mask):
        input_sequence = self.create_input_sequence(root_hidden_state, action_sequence) # Shape: (B, sequence_length, transformer_hidden_size)
        causal_mask = self.create_causal_mask(input_sequence.size(1)).to(input_sequence.device) # Shape: (sequence_length, sequence_length)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_sequence, mask=causal_mask, src_key_padding_mask=action_mask)  # Shape: (B, sequence_length, transformer_hidden_size)

        policy_logits = self.policy_head(transformer_output)  # Shape: (B, sequence_length, action_space_size)
        value = self.value_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)
        reward = self.reward_head(transformer_output)  # Shape: (B, sequence_length, full_support_size)

        return policy_logits, value, reward


    def random_prediction(self, device):
        policy_logits = torch.rand((1, self.action_space_size), device=device, requires_grad=True)
        value = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        reward = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        return policy_logits, value, reward

    def representation(self, observation):
        if self.representation_network_type == "res":
            return self.representation_res(observation)
        elif self.representation_network_type == "mlp":
            return self.representation_mlp(observation)

    def representation_res(self, observation):
        # to device
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

        encoded_state = encoded_state.view(encoded_state.size(0), -1) # (seq_len, batch_size, feature_dim)
        # linear projection
        encoded_state = nn.Linear(encoded_state.size(1), self.transformer_hidden_size,
                                      device=encoded_state.device)(encoded_state)

        return encoded_state

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
            pos_encoding = pos_encoding.expand(embedded_actions.size(0), -1, -1)  # Shape: (B, sequence_length, transformer_hidden_size)
        else:
            # Sinusoidal positional encoding
            pos_encoding = self.positional_encoding[:sequence_length]  # Shape: (sequence_length, transformer_hidden_size)
            pos_encoding = pos_encoding.unsqueeze(0).expand(embedded_actions.size(0), -1, -1)  # Shape: (B, sequence_length, transformer_hidden_size)

        return pos_encoding


    def create_input_sequence(self, root_hidden_state, action_sequence):
        # root hidden state: (B, x)
        # action sequence: (B, y)
        batch_size = root_hidden_state.size(0)

        # Embed the action sequence
        if action_sequence is None:
            # Create an empty embedded_actions tensor
            embedded_actions = torch.empty(batch_size, 0, self.transformer_hidden_size, device=root_hidden_state.device)
        else:
            # Embed the action sequence
            embedded_actions = self.action_embedding(action_sequence)  # Shape: (B, y, transformer_hidden_size)

        # Project the root hidden state to the transformer hidden size
        if self.use_proj:
            root_hidden_state = self.hidden_state_proj(root_hidden_state)
        state_embedding = root_hidden_state.unsqueeze(1)  # Shape: (B, 1, transformer_hidden_size)

        # Total sequence length (including the root hidden state)
        sequence_length = embedded_actions.size(1) + 1  # +1 for the root hidden state

        # Get positional encoding
        pos_encoding = self.get_positional_encoding(sequence_length,
                                                    embedded_actions)  # Shape: (B, y+1 transformer_hidden_size)

        # Construct the input sequence by concatenating the state embedding and action embeddings
        state_action_sequence = torch.cat([state_embedding, embedded_actions], dim=1)  # Shape: (B, y+1, transformer_hidden_size)

        # Add positional encodings
        return state_action_sequence + pos_encoding


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
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
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

        if observation_shape[1] == 1 or observation_shape[2] == 1:
            self.conv_type = "1x1"
        else:
            self.conv_type = "3x3"
        #self.conv_type = "1x1" # todo overwrite

        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels, conv_type = self.conv_type
        )

        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x