import numpy
import torch


class Node:

    def __init__(self, prior, name="root", use_reward=True):
        self.visit_count_ = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.value_evaluation = 0
        self.use_reward = use_reward

        self.ucb_score = None

        # for debugging
        self.name = name

    def expanded(self):
        return len(self.children) > 0

    def make_child(self, prior, child_name):
        """Factory method to create a child node."""
        return Node(prior, name=child_name, use_reward=self.use_reward)

    def get_visit_count(self):
        return self.visit_count_

    def increment_visit_count(self):
        self.visit_count_ += 1

    def value(self): # todo make in mvc format
        if self.get_visit_count() == 0:
            return 0
        return self.value_sum / self.get_visit_count()


    def expand(self, actions, to_play, value , reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward if self.use_reward else 0
        self.value_evaluation = value
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        a_dict = a_dict_cg if len(actions) == 3 else a_dict_ll
        for action, p in policy.items():
            a_name = a_dict[action]
            child_name = f"{self.name}_{a_name}" if self.name != "root" else f"_{a_name}"
            self.children[action] = self.make_child(p, child_name)

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


    def __repr__(self):
        return self.name


class MVCNode(Node):
    def __init__(self, prior, action_space_size, parent=None, name="root", use_reward=True):
        super().__init__(prior, name, use_reward)
        self.parent = parent
        self.variance = None
        self.policy_value = None
        self.action_space_size = action_space_size

    def value(self):
        """
        Override the value method to return the policy value.
        """
        if self.policy_value is None:
            return 0
        return self.policy_value

    def get_visit_count(self):
        raise NotImplementedError("MVCNode does not use visit count")

    def make_child(self, prior, child_name):
        """
        Override the factory method to create a MVCNode child.
        """
        return MVCNode(prior,
                       action_space_size=self.action_space_size,
                       name=child_name,
                       use_reward=self.use_reward,
                       parent=self)

    def reset_var_val(self):
        """
        Reset the variance and policy_value attributes recursively.
        """
        self.variance = None
        self.policy_value = None
        for child in self.children.values():
            child.reset_var_val()

    def reset_var(self):
        """
        Reset the variance attribute recursively.
        """
        self.variance = None
        for child in self.children.values():
            child.reset_var()

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