import torch
from networks.abstract_network import AbstractNetwork
from models import mlp
import torch.nn as nn

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
        transformer_hidden_size=128,
        max_seq_length=50,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.seq_mode = seq_mode

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
        self.dynamics_reward_network = cond_wrap(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = cond_wrap(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = cond_wrap(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

        # Transformer components for value prediction
        self.transformer_hidden_size = transformer_hidden_size
        self.action_embedding = nn.Embedding(action_space_size, transformer_hidden_size)
        self.hidden_state_proj = nn.Linear(encoding_size, transformer_hidden_size)
        self.positional_encoding = nn.Embedding(max_seq_length + 1, transformer_hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_size,
            nhead=transformer_heads,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=transformer_layers,
        )
        self.value_head = nn.Linear(transformer_hidden_size, self.full_support_size)


    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

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

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
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

    def recurrent_inference(self, encoded_state, action, root_hidden_state=None, action_sequence=None):
        assert action_sequence is not None, "Transformer needs an action sequence"
        assert root_hidden_state is not None, "Transformer needs a hidden state"

        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state
