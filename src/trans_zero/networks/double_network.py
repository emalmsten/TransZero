import torch
from trans_zero.networks.abstract_network import AbstractNetwork


class MuZeroDoubleNetwork(AbstractNetwork):
    def __init__(
        self,

        trans_network,
        fully_network

    ):
        super().__init__()
        self.trans_network = trans_network
        self.fulcon_network = fully_network
        #self.representation = trans_network.representation
        self.full_support_size = trans_network.full_support_size


    def initial_inference(self, observation):
        encoded_state = self.fulcon_network.representation(observation)
        policy_logits, value = self.fulcon_network.prediction(encoded_state)

        encoded_state = self.trans_network.representation(observation)
        trans_policy_logits, trans_value, _ = self.trans_network.prediction(encoded_state)

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
            torch.cat([value, trans_value], dim=0),
            torch.cat([reward, reward], dim=0),
            torch.cat([policy_logits, trans_policy_logits], dim=0),
            torch.cat([encoded_state, encoded_state], dim=0),
        )


    def recurrent_inference(self, encoded_state, action, latent_root_state=None, action_sequence=None):

        value, reward, policy_logits, next_encoded_state = self.fulcon_network.recurrent_inference(encoded_state, action)
        trans_value, trans_reward, trans_policy_logits, _ = self.trans_network.recurrent_inference(None, None, latent_root_state, action_sequence)

        return (
                torch.cat([value, trans_value], dim=0),
                torch.cat([reward, trans_reward], dim=0),
                torch.cat([policy_logits, trans_policy_logits], dim=0),
                encoded_state,
            )
