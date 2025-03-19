import torch as th

from value_utils.value_transforms import IdentityValueTransform, ValueTransform




# TODO: can improve this implementation
def policy_value(
    node,
    policy,  # PolicyDistribution | th.distributions.Categorical,
    discount_factor: float,
):
    action_space_size = 3  # todo emil hardcoded

    # return the q value the node with the given policy
    # with the defualt tree evaluator, this should return the same as the default value

    # added false since it cant happen in muzero // Emil
    if False and node.terminal:
        val = th.tensor(node.reward, dtype=th.float32)
        node.policy_value = val
        return val

    if node.policy_value:
        return node.policy_value

    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True, action_space_size=action_space_size)

    probabilities: th.Tensor = pi.probs
    assert probabilities.shape[-1] == int(action_space_size) + 1
    own_propability = probabilities[-1]  # type: ignore
    child_propabilities = probabilities[:-1]  # type: ignore
    child_values = th.zeros_like(child_propabilities, dtype=th.float32)
    for action, child in node.children.items():
        child_values[action] = policy_value(child, policy, discount_factor)

    val = node.reward + discount_factor * (
        own_propability * node.value_evaluation
        + (child_propabilities * child_values).sum()
    )
    node.policy_value = val
    return val


def reward_variance(node):
    return 0.0


def value_evaluation_variance(node):
    # if we want to duplicate the default tree evaluator, we can return 1 / visits
    # In reality, the variance should be lower for terminal nodes
    if False: # todo emil, node.terminal does not exist for muzero:
        return 1.0 / float(node.visits)
    else:
        return 1.0


def independent_policy_value_variance(
    node,
    policy,
    discount_factor: float,
):
    if node.variance is not None:
        return node.variance
    # return the variance of the q value the node with the given policy
    if isinstance(policy, th.distributions.Categorical):
        pi = policy
    else:
        pi = policy.softmaxed_distribution(node, include_self=True)

    probabilities_squared = pi.probs**2  # type: ignore
    own_propability_squared = probabilities_squared[-1]
    child_propabilities_squared = probabilities_squared[:-1]
    child_variances = th.zeros_like(child_propabilities_squared, dtype=th.float32)
    for action, child in node.children.items():
        child_variances[action] = independent_policy_value_variance(
            child, policy, discount_factor
        )

    var = reward_variance(node) + discount_factor**2 * (
        own_propability_squared * value_evaluation_variance(node)
        + (child_propabilities_squared * child_variances).sum()
    )
    node.variance = var
    return var


def get_children_policy_values(
    parent,
    policy, # PolicyDistribution | th.distributions.Categorical,
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,
) -> th.Tensor:
    # have a look at this, infs mess things up
    vals = th.ones(int(parent.action_space.n), dtype=th.float32) * -th.inf
    for action, child in parent.children.items():
        vals[action] = policy_value(child, policy, discount_factor)
    vals = transform.normalize(vals)
    return vals


def get_children_inverse_variances(
    parent, policy, discount_factor: float
) -> th.Tensor:
    inverse_variances = th.zeros(int(parent.action_space.n), dtype=th.float32)
    for action, child in parent.children.items():
        inverse_variances[action] = 1.0 / independent_policy_value_variance(
            child, policy, discount_factor
        )

    return inverse_variances


def get_children_policy_values_and_inverse_variance(
    parent,
    policy, # PolicyDistribution
    discount_factor: float,
    transform: ValueTransform = IdentityValueTransform,
    include_self: bool = False,
) -> tuple[th.Tensor, th.Tensor]:
    """
    This is more efficent than calling get_children_policy_values and get_children_variances separately
    """
    action_space_size = 3 # todo hardcoded
    vals = th.ones(action_space_size + include_self, dtype=th.float32) * -th.inf
    inv_vars = th.zeros_like(vals + include_self, dtype=th.float32)
    for action, child in parent.children.items():
        pi = policy.softmaxed_distribution(child, action_space_size=action_space_size, include_self=True)
        vals[action] = policy_value(child, pi, discount_factor)
        inv_vars[action] = 1 / independent_policy_value_variance(
            child, pi, discount_factor
        )
    if include_self:
        vals[-1] = parent.value_evaluation
        inv_vars[-1] = 1 / value_evaluation_variance(parent)

    normalized_vals = transform.normalize(vals)
    return normalized_vals, inv_vars

def expanded_mask(node) -> th.Tensor:
    mask = th.zeros(int(node.action_space.n), dtype=th.float32)
    mask[node.children] = 1.0
    return mask

def get_children_visits(node) -> th.Tensor:
    action_space_size = 3 # todo hardcoded

    visits = th.zeros(action_space_size, dtype=th.float32)
    for action, child in node.children.items():
        visits[action] = child.visit_count

    return visits

def get_transformed_default_values(node, transform: ValueTransform = IdentityValueTransform) -> th.Tensor:
    vals = th.ones(int(node.action_space.n), dtype=th.float32) * -th.inf
    for action, child in node.children.items():
        vals[action] = child.default_value()

    return transform.normalize(vals)

def puct_multiplier(c: float, node):
    """
    lambda_N from the mcts as policy optimisation paper.
    """
    return c * (node.visits**0.5) / (node.visits + int(node.action_space.n))


def Q_theta_tensor(node, discount: float, transform: ValueTransform = IdentityValueTransform) -> th.Tensor:
    """
    Returns a vector with the Q_theta values. Zero for unvisited nodes, for visited nodes it is return + the discoutned value evaluation of the node
    """
    vals = th.zeros(int(node.action_space.n), dtype=th.float32)
    for action, child in node.children.items():
        val = .0 if child.is_terminal() else child.value_evaluation
        vals[action] = child.reward + discount * val
    return transform.normalize(vals)
