import time
from abc import ABC, abstractmethod
import torch as th

from .value_transforms import IdentityValueTransform, ValueTransform

# todo maybe get some more order of mvc utils

def custom_softmax_old(
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

def custom_softmax(
    probs: th.Tensor,
    temperature: float | None = None,
) -> th.Tensor:
    """Applies AlphaZero-style heuristic temperature scaling to the input tensor.

    Args:
        probs (th.Tensor): Visit counts or unnormalized scores.
        temperature (float): Temperature value. None means use as-is. 0 means argmax.
        action_mask (th.Tensor, optional): Mask for invalid actions.

    Returns:
        th.Tensor: Normalized probability distribution.
    """

    if temperature is None:
        p = probs

    elif temperature == 0.0:
        max_prob = th.max(probs, dim=-1, keepdim=True).values
        p = (probs == max_prob).float()
    else:
        p = probs ** (1.0 / temperature)
        p = p / p.sum(dim=-1, keepdim=True)

        inf_mask = th.isinf(p)
        if inf_mask.any():
            # give each infinite-entry an equal share
            p = inf_mask.to(p.dtype) / inf_mask.sum()
        else:
            # otherwise do the normal normalization
            p = p / p.sum(dim=-1, keepdim=True)

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
        #return int(self.softmaxed_distribution(node).sample().item())
        return node.get_pi().sample().item()

    @abstractmethod
    def _probs(self, node, children_vals, children_inv_vars) -> th.Tensor:
        """
        Returns the relative probabilities of the actions (excluding the special action)
        """
        pass




class MeanVarianceConstraintPolicy(PolicyDistribution):
    """
    Selects the action with the highest inverse variance of the q value.
    Should return the same as the default tree evaluator
    """
    def __init__(self, config, *args, **kwargs):
        # add the self prob arg from config to args

        super().__init__(*args, **kwargs)
        self.beta = config.mvc_beta
        self.discount_factor = config.discount


    def _probs(self, node, children_vals, children_inv_vars) -> th.Tensor:

        # Build unnormalized probabilities (a typical mean-variance approach)
        #   unnorm_action_i = (inv_variance) * exp( beta * normalized_value )


        logits = self.beta * th.nan_to_num(children_vals)
        # check if there are any nans


        # logits - logits.max() is to avoid numerical instability
        probs = children_inv_vars * th.exp(logits - logits.max())

        return probs


    def layer_probs(self, layer, children_vals, children_inv_vars):

        # Build unnormalized probabilities (a typical mean-variance approach)
        #   unnorm_action_i = (inv_variance) * exp( beta * normalized_value )
        logits = self.beta * th.nan_to_num(children_vals)

        # logits - logits.max() is to avoid numerical instability
        probs = children_inv_vars * th.exp(logits - logits.max())

        return probs


