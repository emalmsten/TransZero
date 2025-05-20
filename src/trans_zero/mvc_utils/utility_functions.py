import time

import torch as th

from .value_transforms import IdentityValueTransform, ValueTransform


def policy_value_and_variance(node, discount):
    """
    Calculates the policy value and variance for a node.
    This is used to calculate the value of a node in the tree.
    The value is calculated as the sum of the expected value of the children
    and the expected value of the current node.
    The variance is calculated as the sum of the variances of the children
    and the variance of the current node.
    """
    # todo can move this part until ===== to its own method
    children_vals = node.get_children_vals(include_self=True).squeeze()
    children_vars = node.get_children_vars(include_self=True).squeeze()
    children_inv_vars = node.get_children_inv_vars(include_self=True).squeeze()

    # reclaculation of pi
    raw_pi_probs = node.policy._probs(node, children_vals=children_vals, children_inv_vars=children_inv_vars)
    node.set_pi_probs(raw_pi_probs)
    pi_probs = node.get_pi(include_self=True).probs
    # =======

    reward_variance = 0.0
    value_evaluation_variance = 1.0

    # torch squre for quickness
    probabilities_squared = th.pow(pi_probs, 2)
    assert pi_probs.shape[-1] == node.action_space_size + 1
    own_propability, child_propabilities = pi_probs[-1], pi_probs[:-1]
    own_propability_squared, child_propabilities_squared = probabilities_squared[-1], probabilities_squared[:-1]

    child_values = children_vals[:-1]
    child_variances = children_vars[:-1]

    val = node.get_reward() + discount * (
            own_propability * node.get_value_eval()
            + (child_propabilities * child_values).sum()
    )

    var = reward_variance + discount ** 2 * (
            own_propability_squared * value_evaluation_variance
            + (child_propabilities_squared * child_variances).sum()
    )

    return val, var, pi_probs  # , probabilities



# todo see if can refactor this to use the node version
def policy_value_and_variance_layer(layer, discount):
    children_vals = layer.get_child_layer_vals(include_self=True)
    children_vars = layer.get_child_layer_vars(include_self=True)
    children_inv_vars = layer.get_child_layer_inv_vars(include_self=True)

    pi_probs = layer.subtree.policy.layer_probs(layer, children_vals=children_vals, children_inv_vars=children_inv_vars)

    reward_var = layer.get_reward_vars()
    value_eval_var = layer.get_value_eval_vars()

    probs_squared = th.pow(pi_probs, 2)
    own_prob, child_prob_excl = pi_probs[:, -1:], pi_probs[:, :-1]
    own_prob_squared, child_prob_excl_squared = probs_squared[:, -1:], probs_squared[:, :-1]

    child_vals_excl = children_vals[:, :-1]
    child_vars_excl = children_vars[:, :-1]

    # rewards is (9, 1)
    # own_prob is (4)
    # rews = layer.get_rewards()
    # lay_evals = layer.get_value_evals()
    # todo check step for step at some point
    val = (
            layer.get_rewards()
            + discount * (
                (child_prob_excl * child_vals_excl).sum(dim=1, keepdim=True)  # first sum the children
                + own_prob * layer.get_value_evals() # then add the own prob, same as concatenating and summing over everything
            )
    )

    var = (
            reward_var
            + discount ** 2 * (
                    (child_prob_excl_squared * child_vars_excl).sum(dim=1, keepdim=True)
                    + own_prob_squared * value_eval_var
            )
    )

    return val, var, pi_probs





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

#
# # TODO: can improve this implementation
# def policy_value(
#     node,
#     discount_factor: float,
# ):
#     pi = node.get_pi(include_self=True)
#
#     probabilities: th.Tensor = pi.probs
#     assert probabilities.shape[-1] == node.action_space_size + 1
#
#     own_propability = probabilities[-1]  # type: ignore
#     child_propabilities = probabilities[:-1]  # type: ignore
#
#     child_values = node.children_vals[:-1] #th.empty_like(child_propabilities, dtype=th.float32)
#
#     val = node.reward + discount_factor * (
#         own_propability * node.value_evaluation
#         + (child_propabilities * child_values).sum()
#     )
#
#     return val
#
#
# def independent_policy_value_variance(
#     node,
#     discount_factor: float,
# ):
#
#     pi = node.get_pi(include_self=True)
#
#     probabilities_squared = pi.probs**2  # type: ignore
#     own_propability_squared = probabilities_squared[-1]
#     child_propabilities_squared = probabilities_squared[:-1]
#
#     child_variances = node.children_vars[:-1]
#
#     var = reward_variance(node) + discount_factor**2 * (
#         own_propability_squared * value_evaluation_variance(node)
#         + (child_propabilities_squared * child_variances).sum()
#     )
#
#     return var
