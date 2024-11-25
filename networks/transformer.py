import torch
from networks.abstract_network import AbstractNetwork
from models import mlp
import torch.nn as nn
import math

# currently just a fully connected network
class MuZeroTransformerNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
        seq_mode,

        # todo define in config
        transformer_layers=2,
        transformer_heads=2,
        transformer_hidden_size=16,
        max_seq_length=50,
        positional_embedding_type='sinus',  # sinus or learned
        value_network = "transformer",
        policy_network = "transformer",
        reward_network = "fully_connected",

    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.seq_mode = seq_mode
        self.value_network = value_network
        self.policy_network = policy_network

        def cond_wrap(net):
            return net if self.seq_mode else torch.nn.DataParallel(net)

        self.representation_network = cond_wrap(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = cond_wrap(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )

        if reward_network == "fully_connected":
            self.dynamics_reward_network = cond_wrap(
                mlp(encoding_size, fc_reward_layers, self.full_support_size)
            )

        if self.policy_network == "fully_connected":
            self.prediction_policy_network = cond_wrap(
                mlp(encoding_size, fc_policy_layers, self.action_space_size)
            )

        if self.value_network == "fully_connected":
            self.prediction_value_network = cond_wrap(
                mlp(encoding_size, fc_value_layers, self.full_support_size)
            )

        # Transformer components for value prediction
        self.transformer_hidden_size = transformer_hidden_size
        self.action_embedding = nn.Embedding(action_space_size, transformer_hidden_size)
        self.hidden_state_proj = nn.Linear(encoding_size, transformer_hidden_size)
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
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=transformer_layers,
        )

        if self.policy_network == "transformer":
            self.policy_head = nn.Linear(transformer_hidden_size, self.action_space_size)

        if self.value_network == "transformer":
            self.value_head = nn.Linear(transformer_hidden_size, self.full_support_size)

        if self.reward_network == "transformer":
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


    def prediction(self, encoded_state, action_sequence=None, root_hidden_state=None):
        root_hidden_state = root_hidden_state if root_hidden_state is not None else encoded_state
        rand_policy_logits, rand_value, rand_reward = self.random_prediction(device = encoded_state.device)
        trans_policy_logits, trans_value, trans_reward = self.transformer_prediction(root_hidden_state, action_sequence)
        fully_connected_policy_logits, fully_connected_value, fully_connected_reward = self.fully_connected_prediction(encoded_state)

        policy_dict = {"random": rand_policy_logits, "transformer": trans_policy_logits, "fully_connected": fully_connected_policy_logits}
        val_dict = {"random": rand_value, "transformer": trans_value, "fully_connected": fully_connected_value}
        reward_dict = {"random": rand_reward, "transformer": trans_reward, "fully_connected": fully_connected_reward}

        return policy_dict[self.policy_network], val_dict[self.value_network], reward_dict[self.reward_network]


    def fully_connected_prediction(self, encoded_state):
        policy_logits, value, reward = None, None, None
        if self.policy_network == "fully_connected":
            policy_logits = self.prediction_policy_network(encoded_state)
        if self.value_network == "fully_connected":
            value = self.prediction_value_network(encoded_state)
        if self.reward_network == "fully_connected":
            reward = self.dynamics_reward_network(encoded_state)

        return policy_logits, value, reward


    def transformer_prediction(self, root_hidden_state, action_sequence=None):
        input_sequence = self.create_input_sequence(root_hidden_state, action_sequence)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_sequence)  # Shape: (batch_size, sequence_length, transformer_hidden_size)

        # Obtain the value prediction from the last token's output
        transformer_output_last = transformer_output[:, -1, :]  # Shape: (batch_size, transformer_hidden_size)

        policy_logits, value, reward = None, None, None
        if self.policy_network == "transformer":
            policy_logits = self.policy_head(transformer_output_last) # Shape: (batch_size, action_space_size)
        if self.value_network == "transformer":
            value = self.value_head(transformer_output_last) # Shape: (batch_size, full_support_size)
        if self.reward_network == "transformer":
            reward = self.reward_head(transformer_output_last)

        return policy_logits, value, reward


    def random_prediction(self, device):
        policy_logits = torch.rand((1, self.action_space_size), device=device, requires_grad=True)
        value = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        reward = torch.rand((1, self.full_support_size), device=device, requires_grad=True)
        return policy_logits, value, reward


    def representation(self, observation):
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


    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)

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
            pos_encoding = pos_encoding.expand(embedded_actions.size(0), -1, -1)  # Shape: (batch_size, sequence_length, transformer_hidden_size)
        else:
            # Sinusoidal positional encoding
            pos_encoding = self.positional_encoding[:sequence_length]  # Shape: (sequence_length, transformer_hidden_size)
            pos_encoding = pos_encoding.unsqueeze(0).expand(embedded_actions.size(0), -1, -1)  # Shape: (batch_size, sequence_length, transformer_hidden_size)

        return pos_encoding


    def create_input_sequence(self, root_hidden_state, action_sequence):
        # root hidden state: (batch_size, x)
        # action sequence: (batch_size, y)

        # Embed the action sequence
        if action_sequence is None:
            batch_size = root_hidden_state.size(0)
            # Create an empty embedded_actions tensor
            embedded_actions = torch.empty(batch_size, 0, self.transformer_hidden_size, device=root_hidden_state.device)
        else:
            # Embed the action sequence
            embedded_actions = self.action_embedding(action_sequence)  # Shape: (batch_size, y, transformer_hidden_size)

        # Project the root hidden state to the transformer hidden size
        state_embedding = self.hidden_state_proj(root_hidden_state)  # Shape: (batch_size, transformer_hidden_size)
        state_embedding = state_embedding.unsqueeze(1)  # Shape: (batch_size, 1, transformer_hidden_size)

        # Total sequence length (including the root hidden state)
        sequence_length = embedded_actions.size(1) + 1  # +1 for the root hidden state

        # Get positional encoding
        pos_encoding = self.get_positional_encoding(sequence_length,
                                                    embedded_actions)  # Shape: (batch_size, y+1 transformer_hidden_size)

        # Construct the input sequence by concatenating the state embedding and action embeddings
        state_action_sequence = torch.cat([state_embedding, embedded_actions], dim=1)  # Shape: (batch_size, y+1, transformer_hidden_size)

        # Add positional encodings
        return state_action_sequence + pos_encoding



    def recurrent_inference(self, encoded_state, action, root_hidden_state=None, action_sequence=None):
        assert action_sequence is not None, "Transformer needs an action sequence"
        assert root_hidden_state is not None, "Transformer needs a hidden state"

        next_encoded_state = self.dynamics(encoded_state, action)

        policy_logits, value, reward = self.prediction(next_encoded_state, action_sequence, root_hidden_state)

        return value, reward, policy_logits, next_encoded_state
