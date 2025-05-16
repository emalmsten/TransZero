import numpy
import torch
import torch as th

from trans_zero.mvc_utils.policies import MeanVarianceConstraintPolicy, custom_softmax, custom_softmax_old
from trans_zero.mvc_utils.utility_functions import policy_value, independent_policy_value_variance, \
    policy_value_and_variance


class Node:

    def __init__(self, prior, config, parent=None, name="root"):
        self.config = config
        self.action_space_size = len(self.config.action_space)
        self.use_reward = config.predict_reward
        self.parent = parent

        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.value_evaluation = 0

        self.ucb_score = None
        self.is_leaf = True

        # for debugging
        self.name = name

    def expanded(self):
        return len(self.children) > 0

    def make_child(self, prior, child_name, action): # todo
        """Factory method to create a child node."""
        self.is_leaf = False
        return Node(prior, self.config, name=child_name, parent=self)

    def get_visit_count(self):
        return self.visit_count

    def increment_visit_count(self):
        self.visit_count += 1

    def get_value(self):
        if self.get_visit_count() == 0:
            return 0
        return self.value_sum / self.get_visit_count()


    def expand(self, available_actions, to_play, value, reward, policy_values, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward if self.use_reward else 0

        self.value_evaluation = value
        self.add_val_and_var() # reconsider place for this

        self.hidden_state = hidden_state
        assert len(self.children) == 0, f"{self.name}: expanding already expanded node"

        for action, p in zip(available_actions, policy_values):
            self.children[action] = self.make_child(p, self.get_child_name(action), action)


    def add_val_and_var(self):
        pass
        #raise NotImplementedError("This method should be overridden in subclasses")

    def get_child_name(self, action):
        return f"{self.name}_{action}" if self.name != "root" else f"r_{action}"

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


    def recalculate_val_and_var(self):
        pass
        #raise NotImplementedError("This method should be overridden in subclasses")

    def set_children_val_and_vars(self, child):
        pass
        #raise NotImplementedError("This method should be overridden in subclasses")

    def print_tree(self, level=0):
        print(" " * (level * 4) + f"O")
        for child in self.children.values():
            child.print_tree(level + 1)

    def __repr__(self):
        return self.name


class MVCNode(Node):
    def __init__(self, prior, config, parent=None, name="root", action = None):
        super().__init__(prior, config, name)
        self.config = config
        self.parent = parent

        self.action = action

        self.variance = None
        self.inv_var = None
        self.policy_value = None
        self.pi_probs = None

        self.policy = parent.policy if parent is not None else MeanVarianceConstraintPolicy(config)
        self.name = name

        self.children = {}

        # fill with self.config.discount**2
        self.children_vars = torch.full(
            (self.action_space_size + 1,),
            self.config.discount**2,
            dtype=torch.float32,
        )
        self.children_vals = torch.zeros(self.action_space_size + 1)
        self.children_inv_vars = torch.full(
            (self.action_space_size + 1,),
            1 / self.config.discount**2,
            dtype=torch.float32,
        )
        self.children_inv_vars[-1] = 1.0
        self.children_vars[-1] = 1.0

    def recalculate_val_and_var(self):
        self.reset_all()
        self.policy_value, self.variance = policy_value_and_variance(self, self.config.discount)

    def get_value(self):
        """
        Override the value method to return the policy value.
        """
        # return 0.5
        if self.policy_value is None:
            if self.is_leaf:
                self.policy_value = 0.0
            else:
                raise ValueError("Policy value is None, please call get_policy_value() first")
                #self.policy_value = policy_value(self, self.policy.discount_factor)

        return self.policy_value

    def set_children_val_and_vars(self, child):
        self.children_vals[child.action] = child.policy_value
        self.children_vars[child.action] = child.variance
        self.children_inv_vars[child.action] = 1 / child.variance


    def add_val_and_var(self):
        self.children_vals[-1] = self.value_evaluation

    def get_variance(self):
        if self.variance is None:
            if self.is_leaf:
                self.variance = self.config.discount**2
            else:
                raise ValueError("variance is None, please call get_variance() first")
                # self.variance = independent_policy_value_variance(
                #     self, self.policy.discount_factor
                # )

        return self.variance


    def get_inv_var(self):
        if self.inv_var is None:
            self.inv_var = 1.0 / self.get_variance()

        return self.inv_var


    def get_pi(self, include_self=False, temperature = None):
        """
        Get the policy distribution for the node.
        """
        if self.pi_probs is None:
            if not self.is_leaf:
                self.pi_probs = self.policy._probs(self)
            else:
                self.pi_probs = th.zeros(self.action_space_size + 1, dtype=th.float32)
                self.pi_probs[-1] = 1.0

        pi_probs = self.pi_probs if include_self else self.pi_probs[:-1]

        if temperature is not None:
            if self.config.use_old_softmax:
                pi_probs = custom_softmax_old(pi_probs, temperature)
            else:
                pi_probs = custom_softmax(pi_probs, temperature)


        return th.distributions.Categorical(probs=pi_probs)


    def get_visit_count(self):
        return self.visit_count
        #todo raise NotImplementedError("MVCNode does not use visit count")


    def make_child(self, prior, child_name, action):
        """
        Override the factory method to create a MVCNode child.
        """
        self.is_leaf = False
        return MVCNode(prior, self.config, name=child_name, parent=self, action=action)


    def reset_all(self):
        """
        Reset the variance attribute recursively.
        """

        self.variance = None
        self.inv_var = None
        self.policy_value = None
        self.pi_probs = None


    def __repr__(self):
        return self.name


# todo clean
a_dict_cg = {
    0: 'L',
    1: 'R',
    2: 'F'
}

a_dict_ll = {
    0: 'N',
    1: 'L',
    2: 'M',
    3: 'R'
}