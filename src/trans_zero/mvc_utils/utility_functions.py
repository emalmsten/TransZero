import time

import torch as th

from .value_transforms import IdentityValueTransform, ValueTransform


def policy_value_and_variance(node, discount_factor: float):
    """
    Calculates the policy value and variance for a node.
    This is used to calculate the value of a node in the tree.
    The value is calculated as the sum of the expected value of the children
    and the expected value of the current node.
    The variance is calculated as the sum of the variances of the children
    and the variance of the current node.
    """

    pi = node.get_pi(include_self=True)

    probabilities: th.Tensor = pi.probs
    # torch squre for quickness
    probabilities_squared = th.pow(probabilities, 2)
    assert probabilities.shape[-1] == node.action_space_size + 1
    own_propability, child_propabilities = probabilities[-1], probabilities[:-1]
    own_propability_squared, child_propabilities_squared = probabilities_squared[-1], probabilities_squared[:-1]

    child_values = node.children_vals[:-1]
    child_variances = node.children_vars[:-1]

    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )

    var = reward_variance(node) + discount_factor**2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )

    return val, var #, probabilities



# TODO: can improve this implementation
def policy_value(
    node,
    discount_factor: float,
):
    pi = node.get_pi(include_self=True)

    probabilities: th.Tensor = pi.probs
    assert probabilities.shape[-1] == node.action_space_size + 1

    own_propability = probabilities[-1]  # type: ignore
    child_propabilities = probabilities[:-1]  # type: ignore

    child_values = node.children_vals[:-1] #th.empty_like(child_propabilities, dtype=th.float32)

    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )

    return val


def independent_policy_value_variance(
    node,
    discount_factor: float,
):

    pi = node.get_pi(include_self=True)

    probabilities_squared = pi.probs**2  # type: ignore
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]

    child_variances = node.children_vars[:-1]

    var = reward_variance(node) + discount_factor**2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )

    return var


def reward_variance(node):
    # todo, need to figure out the reward variance for deterministic environments
    return 0.0


def value_evaluation_variance(node):
    # if we want to duplicate the default tree evaluator, we can return 1 / visits
    # In reality, the variance should be lower for terminal nodes
    # if False: # if node.terminal # todo emil, node.terminal does not exist for muzero:
    #     return 1.0 / float(node.visits)
    # else:
    return 1.0



def get_children_policy_values_and_inverse_variance(
    parent,
    #transform: ValueTransform = IdentityValueTransform,
) -> tuple[th.Tensor, th.Tensor]:
    """
    This is more efficent than calling get_children_policy_values and get_children_variances separately
    """

    return parent.children_vals, parent.children_inv_vars




 # todo unused -----------------------



# def get_children_policy_values(
#     parent,
#     transform: ValueTransform = IdentityValueTransform,
# ) -> th.Tensor:
#     # have a look at this, infs mess things up
#     vals = th.ones(int(parent.action_space.n), dtype=th.float32) * -th.inf
#     for action, child in parent.children.items():
#         vals[action] = child.get_value()
#     vals = transform.normalize(vals)
#     return vals

# def get_children_inverse_variances(
#     parent
# ) -> th.Tensor:
#     inverse_variances = th.zeros(parent.action_space_size, dtype=th.float32)
#     for action, child in parent.children.items():
#         inverse_variances[action] = child.get_inv_var()
#
#     return inverse_variances


def expanded_mask(node) -> th.Tensor:
    mask = th.zeros(int(node.action_space.n), dtype=th.float32)
    mask[node.children] = 1.0
    return mask


def get_children_visits(node) -> th.Tensor:

    visits = th.zeros(node.action_space_size, dtype=th.float32)
    for action, child in node.children.items():
        visits[action] = child.visit_count

    return visits


def get_transformed_default_values(node, transform: ValueTransform = IdentityValueTransform) -> th.Tensor:
    vals = th.ones(int(node.action_space.n), dtype=th.float32) * -th.inf
    for action, child in node.children.items():
        vals[action] = child.default_value()

    return transform.normalize(vals)


