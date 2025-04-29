import numpy
import torch as th

from trans_zero.mvc_utils.policies import MeanVarianceConstraintPolicy, custom_softmax
from trans_zero.mvc_utils.utility_functions import policy_value, independent_policy_value_variance


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

        # for debugging
        self.name = name

    def expanded(self):
        return len(self.children) > 0

    def make_child(self, prior, child_name):
        """Factory method to create a child node."""
        return Node(prior, self.config, name=child_name, parent=self)

    def get_visit_count(self):
        return self.visit_count

    def increment_visit_count(self):
        self.visit_count += 1

    def get_value(self): # todo make in mvc format
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
        self.hidden_state = hidden_state

        for action, p in zip(available_actions, policy_values):
            self.children[action] = self.make_child(p, self.get_child_name(action))


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


    def reset_ucb(self):
        self.ucb_score = None
        for child in self.children.values():
            child.reset_ucb()

    def reset_var(self):
        pass

    def reset_val(self):
        pass

    def reset_pi(self):
        pass


    def __repr__(self):
        return self.name


class MVCNode(Node):
    def __init__(self, prior, config, parent=None, name="root"):
        super().__init__(prior, config, name)
        self.config = config
        self.parent = parent
        self.variance = None
        self.policy_value = None
        self.pi_probs = None

        self.policy = parent.policy if parent is not None else MeanVarianceConstraintPolicy(config)
        self.name = name

        self.prev_variance = None
        self.prev_policy_value = None

        self.diff_var_val = 0
        self.same_var_val = 0

        self.children = {}

    def get_value(self):
        """
        Override the value method to return the policy value.
        """
        # return 0.5
        if self.policy_value is None:
            if self.children:
                self.policy_value = policy_value(self, self.policy.discount_factor)
            else:
                self.policy_value = 0.0

        return self.policy_value


    def get_variance(self):
        if self.variance is None:
            if self.children:
                self.variance = independent_policy_value_variance(
                    self, self.policy.discount_factor
                )
            else:
                self.variance = self.config.discount**2

        return self.variance


    def get_pi(self, include_self=False, temperature = None):
        """
        Get the policy distribution for the node.
        """
        if self.pi_probs is None:
            if self.children:
                self.pi_probs = self.policy.softmaxed_distribution(self)
            else:
                self.pi_probs = th.zeros(self.action_space_size + 1, dtype=th.float32)
                self.pi_probs[-1] = 1.0


        pi_probs = self.pi_probs.clone()

        if not include_self:
            pi_probs = pi_probs[:-1]

        if temperature is not None:
            pi_probs = custom_softmax(pi_probs, temperature, None)

        return th.distributions.Categorical(probs=pi_probs)


    def get_visit_count(self):
        return self.visit_count
        #todo raise NotImplementedError("MVCNode does not use visit count")


    def make_child(self, prior, child_name):
        """
        Override the factory method to create a MVCNode child.
        """
        return MVCNode(prior, self.config, name=child_name, parent=self)


    def reset_var(self):
        """
        Reset the variance attribute recursively.
        """

        self.prev_variance = self.variance
        self.variance = None


    def reset_val(self):
        """
        Reset the policy_value attribute recursively.
        """

        self.prev_variance = self.variance
        self.variance = None


    def reset_pi(self):
        """
        Reset the pi attribute.
        """

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