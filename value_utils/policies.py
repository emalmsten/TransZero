from abc import ABC, abstractmethod
import torch as th
import numpy as np
from value_utils.node import Node
from value_utils.value_transforms import IdentityValueTransform, ValueTransform
from value_utils.utility_functions import get_children_policy_values_and_inverse_variance


def custom_softmax(
    probs: th.Tensor,
    temperature: float | None = None,
    action_mask: th.Tensor | None = None,
) -> th.Tensor:
    """Applies softmax to the input tensor with a temperature parameter.

    Args:
        probs (th.Tensor): Relative probabilities of actions.
        temperature (float): The temperature parameter. None means dont apply softmax. 0 means stochastic argmax.
        action_mask (th.Tensor, optional): A mask tensor indicating which actions are valid to take. The probability of these should be zero.

    Returns:
        th.Tensor: Probs after applying softmax.
    """

    if temperature is None:
        # no softmax
        p = probs

    elif temperature == 0.0:
        max_prob = th.max(probs, dim=-1, keepdim=True).values
        p = (probs == max_prob).float()
    else:
        p = th.nn.functional.softmax(probs / temperature, dim=-1)

    if action_mask is not None:
        p[~action_mask] = 0.0

    return p


class Policy(ABC):
    def __call__(self, node) -> int:
        return self.sample(node)

    @abstractmethod
    def sample(self, node) -> int:
        """Take a node and return an action"""


class PolicyDistribution(Policy):
    """Also lets us view the full distribution of the policy, not only sample from it.
    When we have the distribution, we can choose how to sample from it.
    We can either sample stochasticly from distribution or deterministically choose the action with the highest probability.
    We can also apply softmax with temperature to the distribution.
    """

    temperature: float
    value_transform: ValueTransform

    def __init__(
        self,
        temperature: float = None,
        value_transform: ValueTransform = IdentityValueTransform,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.value_transform = value_transform

    def sample(self, node) -> int:
        """
        Returns a random action from the distribution
        """
        return int(self.softmaxed_distribution(node).sample().item())

    @abstractmethod
    def _probs(self, node) -> th.Tensor:
        """
        Returns the relative probabilities of the actions (excluding the special action)
        """
        pass

    def self_prob(self, node, probs: th.Tensor) -> float:
        """
        Returns the relative probability of selecting the node itself
        """
        return probs.sum() / (node.visit_count - 1)

    def add_self_to_probs(self, node, probs: th.Tensor) -> th.Tensor:
        """
        Takes the current policy and adds one extra value to it, which is the probability of selecting the node itself.
        Should return a tensor with one extra value at the end
        The default choice is to set it to 1/visits
        Note that policy is not yet normalized, so we can't just add 1/visits to the last value
        """
        self_prob = self.self_prob(node, probs)
        return th.cat([probs, th.tensor([self_prob])])

    def softmaxed_distribution(
        self, node, include_self=False, **kwargs
    ) -> th.distributions.Categorical:
        """
        Relative probabilities with self handling
        """
        # policy for leaf nodes
        # note to emil, leaf node handling, not really relevant for now
        if include_self and len(node.children) == 0:
            probs = th.zeros(node.action_space_size + include_self, dtype=th.float32)
            probs[-1] = 1.0
            return th.distributions.Categorical(probs=probs)

        probs = self._probs(node)
        # softmax the probs
        softmaxed_probs = custom_softmax(probs, self.temperature, None)
        if include_self:
            softmaxed_probs = self.add_self_to_probs(node, softmaxed_probs)
        return th.distributions.Categorical(probs=softmaxed_probs)


class RandomPolicy(Policy):
    def sample(self, node) -> int:
        return node.action_space.sample()



class MinimalVarianceConstraintPolicy(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = config.mvc_beta
        self.discount_factor = config.discount

    def get_beta(self):
        return self.beta

    def _probs(self, node) -> th.Tensor:

        normalized_vals, inv_vars = get_children_policy_values_and_inverse_variance(node, self, self.discount_factor, self.value_transform)
        # for action in node.children:
        #     probs[action] = th.exp(beta * (normalized_vals[action] - normalized_vals.max())) * inv_vars[action]
        logits = self.beta * th.nan_to_num(normalized_vals)
        # logits - logits.max() is to avoid numerical instability
        probs = inv_vars * th.exp(logits - logits.max())
        return probs

    def self_prob(self, node: Node, probs: th.Tensor) -> float:
        # todo emil, just a guess
        return 0
        """
        Returns the relative probability of selecting the node itself,
        under the same 'exp(beta * value) * inv_var' scheme used for children.
        """
        # 1) Get children + self values and inv_vars
        normalized_vals, inv_vars = get_children_policy_values_and_inverse_variance(
            parent=node,
            policy=self,
            discount_factor=self.discount_factor,
            transform=self.value_transform,
            include_self=True
        )

        # 2) Compute logits for *all* actions + self
        beta = self.get_beta(node)
        logits = beta * th.nan_to_num(normalized_vals)

        # 3) Avoid numerical instability by shifting by max(logits)
        logits -= logits.max()

        # 4) Convert logits to unnormalized probabilities via exp, then multiply by inv_vars
        unnormalized_probs = inv_vars * th.exp(logits)

        # 5) The parent's "self" score is the last index, so normalize and extract
        parent_score = unnormalized_probs[-1]
        parent_prob = parent_score / unnormalized_probs.sum()

        return float(parent_prob)

