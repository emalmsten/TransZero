import numpy
import torch
import torch as th

from trans_zero.mvc_utils.policies import MeanVarianceConstraintPolicy, mz_normalizing, custom_softmax

import itertools

from trans_zero.mvc_utils.utility_functions import policy_value_and_variance_layer, policy_value_and_variance


class Node:

    def __init__(self, prior, config, parent=None, name="root"):
        self.config = config
        self.action_space_size = len(self.config.action_space)
        # lsit of actions
        self.action_space = self.config.action_space

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
        self.add_own_val() # reconsider place for this

        self.hidden_state = hidden_state
        assert len(self.children) == 0, f"{self.name}: expanding already expanded node"

        for action, p in zip(available_actions, policy_values):
            self.children[action] = self.make_child(p, self.get_child_name(action), action)


    def add_own_val(self):
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

    def get_child(self, action):
        """
        Get the child of the node.
        """
        return self.children[action]

    def get_reward(self):
        return self.reward

    def get_value_eval(self):
        return self.value_evaluation


class MVCNode(Node):
    def __init__(self, prior, config, parent=None, name="root", action = None):
        super().__init__(prior, config, name)
        self.parent = parent

        self.action = action
        self.discount_factor = config.discount

        self.variance = self.discount_factor**2
        self.policy_value = 0.0
        self.pi_probs = None

        self.policy = parent.policy if parent is not None else MeanVarianceConstraintPolicy(config)

        self.children = {}

        # fill with self.config.discount**2
        self.children_vars = torch.full(
            (self.action_space_size + 1,),
            self.discount_factor**2,
            dtype=torch.float32,
        )
        self.children_vals = torch.zeros(self.action_space_size + 1)
        self.children_inv_vars = torch.full(
            (self.action_space_size + 1,),
            1 / self.discount_factor**2,
            dtype=torch.float32,
        )

        # value evaluation variance is always 1.0 in muzero
        self.children_inv_vars[-1] = 1.0
        self.children_vars[-1] = 1.0


    def recalculate_val_and_var(self):
        self.reset_all()
        # todo note that pi probs is cached in that method
        self.policy_value, self.variance= policy_value_and_variance(self, self.discount_factor)
        return self.policy_value, self.variance


    def get_value(self):
        return self.policy_value


    def set_children_val_and_vars(self, child):
        self.children_vals[child.action] = child.policy_value
        self.children_vars[child.action] = child.variance
        self.children_inv_vars[child.action] = 1 / child.variance


    def add_own_val(self):
        self.children_vals[-1] = self.value_evaluation

    def get_variance(self):
        return self.variance

    def get_inv_var(self):
        return 1 / self.variance

    def get_pi_probs(self, include_self=False):
        return self.pi_probs if include_self else self.pi_probs[:-1]

    def set_pi_probs(self, raw_pi_probs):
        """
        Set the policy distribution for the node.
        """
        self.pi_probs = raw_pi_probs

    def get_pi(self, include_self=False, temperature = None):
        """
        Get the policy distribution for the node.
        """
        pi_probs = self.get_pi_probs(include_self)

        if temperature is not None:
            if self.config.use_softmax:
                pi_probs = custom_softmax(pi_probs, temperature)
            else:
                pi_probs = mz_normalizing(pi_probs, temperature)

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
        self.policy_value = None
        self.pi_probs = None



    def get_children_vals(self, include_self=True):
        """
        Get the values of the children of the node.
        """
        return self.children_vals if include_self else self.children_vals[:-1]


    def get_children_vars(self, include_self=True):
        """
        Get the variances of the children of the node.
        """
        return self.children_vars if include_self else self.children_vars[:-1]


    def get_children_inv_vars(self, include_self=True):
        """
        Get the inverse variances of the children of the node.
        """
        return self.children_inv_vars if include_self else self.children_inv_vars[:-1]



    def __repr__(self):
        return self.name



class SubTreeNode():
    """
    SubTreeNode is a special node that is used to represent the PLL (Permutation, Location, and Layer) of the grid.
    It is used in the MVC algorithm to represent the state of the grid.
    """

    def __init__(self, subtree, idx):
        self.subtree = subtree
        assert subtree is not None, "SubTreeNode must have a subtree"
        self.idx = idx

        self.is_in_final_layer = subtree.is_in_final_layer(idx)

        if not self.is_in_final_layer:
            self.children_start_idx, self.children_end_idx = subtree.get_children_indices(idx)
            self.children_slice = slice(self.children_start_idx, self.children_end_idx)


    def __getattr__(self, name):
        # todo add better error messages
        # just get these attr from the subtree
        if name.startswith("get_children"):
            attr = name[13:]
            return lambda: self.subtree.get_list_slice(attr, self.children_slice)

        elif name.startswith("get_"):
            attr = name[4:] + 's' # add the s since you get from a list and they always end with an s
            return lambda: self.subtree.get_list_idx(attr, self.idx)

        # if it does not have the attribute, look in subtree
        try:
            return getattr(self.subtree, name)
        except AttributeError:
            raise AttributeError(f"Neither SubTree nor {type(self).__name__} has the attribute {name}")

        #raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def get_value(self):
        """
        Get the value of the node.
        """
        return self.get_val() # todo


    @property
    def parent(self):
        if self.idx == 0:
            if self.subtree.is_root:
                return None
            else:
                # get the parent node of this subtree (is the same* node as the root of this tree),
                # then return its parent. *same values etc, but different object
                return self.subtree.parent.parent
        else:
            return SubTreeNode(self.subtree, self.subtree.get_parent_idx(self.idx))


    def get_child(self, action):
        if self.is_in_final_layer:
            subtree = self.subtree.get_child_tree(self.idx)
            return SubTreeNode(subtree, action)
        else:
            idx = self.subtree.get_child_idx(self.idx, action)
            return SubTreeNode(self.subtree, idx)

    def get_pi_probs(self, include_self=False):
        """
        Get the policy distribution for the node.
        """
        pi_probs = self.subtree.pi_probs[self.idx]
        return pi_probs if include_self else pi_probs[:-1]

    # todo clean up duplication
    def get_children_vals(self, include_self):
        """
        Get the values of the children of the node.
        """
        if self.is_in_final_layer:
            child_vals = self.subtree.leaf_vals[:self.action_space_size]
        else:
            # todo reconsider renaming method to be able to use get_children_vals()
            child_vals = self.subtree.get_list_slice("vals", self.children_slice)

        if include_self:
            own_val = self.get_value_eval().unsqueeze(0)
            # concat as last column of child_vals
            child_vals = th.cat((child_vals, own_val), dim=0)

        return child_vals


    def get_children_vars(self, include_self):
        """
        Get the variances of the children of the node.
        """
        if self.is_in_final_layer:
            child_vars = self.subtree.leaf_vars[:self.action_space_size]
        else:
            child_vars = self.subtree.get_list_slice("vars", self.children_slice)

        if include_self:
            own_var = self.get_value_eval_var().unsqueeze(0)
            # concat as last column of child_vars
            child_vars = th.cat((child_vars, own_var), dim=0)

        return child_vars


    def get_children_inv_vars(self, include_self):
        """
        Get the inverse variances of the children of the node.
        """
        if self.is_in_final_layer:
            child_inv_vars = self.subtree.leaf_inv_vars[:self.action_space_size]
        else:
            child_inv_vars = self.subtree.get_list_slice("inv_vars", self.children_slice)

        if include_self:
            own_inv_var = self.get_value_eval_var().unsqueeze(0)
            # concat as last column of child_vars
            child_inv_vars = th.cat((child_inv_vars, own_inv_var), dim=0)

        return child_inv_vars

    def set_pi_probs(self, raw_pi_probs):
        """
        Set the policy distribution for the node.
        """
        # todo make set attr for this
        self.subtree.pi_probs[self.idx] = raw_pi_probs


    def get_pi(self, include_self=False, temperature=None):
        """
        # todo is exact same as node, consider refactoring
        Get the policy distribution for the node.
        """
        pi_probs = self.get_pi_probs(include_self)

        if temperature is not None:
            if self.subtree.config.use_softmax:
                pi_probs = custom_softmax(pi_probs, temperature)
            else:
                pi_probs = mz_normalizing(pi_probs, temperature)

        return th.distributions.Categorical(probs=pi_probs)



    def expanded(self):
        return self.subtree.expanded(self.idx)


    def recalculate_val_and_var(self):
        val, var = policy_value_and_variance(self, self.discount)
        self.set_val_and_vars(val, var)
        return val, var


    def set_val_and_vars(self, val, var):
        self.subtree.vals[self.idx] = val
        self.subtree.vars[self.idx] = var
        self.subtree.inv_vars[self.idx] = 1 / var
        # self.subtree.pi_probs[self.idx] = pi_probs # set in policy_value_and_variance

        return val






class SubTreeLayer:
    """
    SubTreeLayer is a special node that is used to represent the PLL (Permutation, Location, and Layer) of the grid.
    It is used in the MVC algorithm to represent the state of the grid.
    """

    def __init__(self, subtree, layer_number):
        self.subtree = subtree
        self.layer_number = layer_number

        self.start, self.end = self.subtree.get_layers_indices(layer_number)
        self.layer_slice = slice(self.start, self.end)

        self.is_final_layer = self.subtree.num_layers == layer_number
        if not self.is_final_layer:
            self.child_start_idx, self.child_end_idx = self.subtree.get_layers_indices(layer_number + 1)
            self.child_layer_slice = slice(self.child_start_idx, self.child_end_idx)

    # used as getter for the ones not below
    def __getattr__(self, name):
        if name.startswith("get_children_"):
            attr = name[13:]
            n = self.subtree.action_space_size
            return lambda: self.subtree.get_list_slice(attr, self.child_layer_slice).reshape(-1, n)
        elif name.startswith("get_"):
            attr = name[4:]
            return lambda: self.subtree.get_list_slice(attr, self.layer_slice)

        try:
            return getattr(self.subtree, name)
        except AttributeError:
            raise AttributeError(f"Neither SubTree nor {type(self).__name__} has the attribute {name}")



    def get_child_layer_vals(self, include_self):
        """
        Get the values of the children of the node.
        """
        if self.is_final_layer:
            child_vals = self.subtree.leaf_vals
        else:
            child_vals = self.get_children_vals()

        if include_self:
            own_vals = self.get_value_evals()
            # concat as last column of child_vals
            child_vals = th.cat((child_vals, own_vals), dim=1)

        return child_vals



    def get_child_layer_vars(self, include_self):
        """
        Get the variances of the children of the node.
        """
        if self.is_final_layer:
            child_vars = self.subtree.leaf_vars
        else:
            child_vars = self.get_children_vars()

        if include_self:
            own_vars = self.get_value_eval_vars()
            # concat as last column of child_vars
            child_vars = th.cat((child_vars, own_vars), dim=1)

        return child_vars


    def get_child_layer_inv_vars(self, include_self):
        """
        Get the variances of the children of the node.
        """
        if self.is_final_layer:
            child_inv_vars = self.subtree.leaf_inv_vars
        else:
            child_inv_vars = self.get_children_inv_vars()

        if include_self:
            own_inv_vars = self.get_value_eval_vars() # same as var since 1/1 = 1
            # concat as last column of child_vars
            child_inv_vars = th.cat((child_inv_vars, own_inv_vars), dim=1)

        return child_inv_vars



    def set_val_and_vars(self, val, var):
        self.subtree.vals[self.layer_slice] = val
        self.subtree.vars[self.layer_slice] = var
        self.subtree.inv_vars[self.layer_slice] = 1.0 / var
        #self.subtree.pi_probs[self.layer_slice] = pi_probs # set in policy_value_and_variance_layer


    def set_pi_probs(self, raw_pi_probs):
        """
        Set the policy distribution for the node.
        """
        # todo make set attr for this
        self.subtree.pi_probs[self.layer_slice] = raw_pi_probs


    def get_pi_layer(self, include_self=False, temperature=None):
        """
        # todo is exact same as node, consider refactoring
        Get the policy distribution for the node.
        """
        pi_probs = self.get_pi_probs()

        if temperature is not None or include_self is False:
            raise NotImplementedError("Temperature nro include self is not implemented for SubTreeLayer")
            if self.subtree.config.use_softmax:
                pi_probs = custom_softmax(pi_probs, temperature)
            else:
                pi_probs = mz_normalizing(pi_probs, temperature)

        return th.distributions.Categorical(probs=pi_probs)



class SubTree():
    """
    PLLNode is a special node that is used to represent the PLL (Permutation, Location, and Layer) of the grid.
    It is used in the MVC algorithm to represent the state of the grid.
    """

    def __init__(self, parent = None, config=None, device=None):

        self.is_root = parent is None
        self.parent = parent

        if self.is_root:
            self.config, self.device = config, device

            self.discount = config.discount
            self.action_space = config.action_space
            self.action_space_size = len(self.action_space)
            self.num_layers = self.config.subtree_layers
            assert self.num_layers > 0, f"num_layers must be greater than 0, but got {self.num_layers}"

            self.policy = MeanVarianceConstraintPolicy(config)
            self.seqs, self.positions, self.action_seq, self.mask = None, None, None, None
            self.size = (self.action_space_size ** (self.num_layers + 1) - 1) // (self.action_space_size - 1) # geometric series of size
            self.non_leaf_size = (self.action_space_size ** self.num_layers) // (self.action_space_size - 1) # geometric series of size

            final_layer_size = self.action_space_size ** self.num_layers
            self.leaf_vars = torch.full((final_layer_size, self.action_space_size), self.discount**2, dtype=torch.float32, device=self.device)
            self.leaf_inv_vars = torch.full((final_layer_size, self.action_space_size), 1 / self.discount**2, dtype=torch.float32, device=self.device)
            self.leaf_vals = torch.zeros((final_layer_size, self.action_space_size), dtype=torch.float32, device=self.device)
            self.value_eval_vars = torch.ones((self.size, 1), dtype=torch.float32, device=self.device)
            self.reward_vars = torch.zeros((self.size, 1), dtype=torch.float32, device=self.device)

        else:
            # for attr in ['config', 'device', 'policy', 'seqs', 'positions', 'action_seq', 'mask',
            #              'leaf_vars', 'leaf_inv_vars', 'leaf_vals']:
            #     setattr(self, attr, getattr(parent_tree, attr))
            parent_tree = self.parent.subtree
            parent_tree.set_child_tree(self, self.parent.idx)

            #
            # def __getattr__(self, item): takes care of the rest of the attributes

        self.child_trees = [None] * (self.action_space_size ** self.num_layers)
        self.value_evals, self.rewards, self.all_policy_probs, self.root_hidden_state = None, None, None, None

        self.vals = torch.empty((self.size, 1), dtype = torch.float32,device=self.device)
        self.inv_vars = torch.empty((self.size, 1), dtype=torch.float32,device=self.device)
        self.vars = torch.empty((self.size, 1), dtype=torch.float32,device=self.device)
        # size =
        self.pi_probs = torch.empty((self.size, self.action_space_size + 1), dtype=torch.float32,device=self.device)


    def __getattr__(self, item):
        try:
            return getattr(self.parent.subtree, item)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {item}")


    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        noise = numpy.random.dirichlet([dirichlet_alpha] * self.action_space_size)
        frac = exploration_fraction
        # for a, n in zip(actions, noise):
        #     self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
        self.prior[0] = self.prior[0] * (1 - frac) + noise * frac


    def set_val_and_var_probs_parent(self):
        """
        Set the value and variance of the parent node.
        """
        self.parent.set_val_and_vars(self.vals[0], self.vars[0])
        self.parent.set_pi_probs(self.pi_probs[0])


    def get_list_slice(self, attr, range_slice):
        return getattr(self, attr)[range_slice]


    def get_list_idx(self, attr, idx):
        return getattr(self, attr)[idx]


    def get_children_indices(self, idx):
        """
        Get the indices of the children of the node.
        """
        b = self.action_space_size
        start_idx = (idx * b) + 1
        end_idx = (idx + 1) * b + 1

        return start_idx, end_idx


    def get_child_idx(self, idx, action):
        """
        Get the child index of the node.
        """
        b = self.action_space_size
        idx = (idx * b) + 1 + action

        return idx


    def get_parent_idx(self, idx):
        """
        Get the parent index of the node.
        """
        b = self.action_space_size
        idx = (idx - 1) // b

        return idx


    def expanded(self, idx):
        if not self.is_in_final_layer(idx):
            return True

        return self.get_child_tree(idx) is not None

    def get_child_tree(self, idx):
        """
        Get the child tree of the node.
        """
        return self.child_trees[idx - self.non_leaf_size]


    def set_child_tree(self, child_tree, idx):
        """
        Set the child tree of the node.
        """
        self.child_trees[idx - self.non_leaf_size] = child_tree


    def is_in_final_layer(self, idx):
        """
        Check if the node is in the final layer of the tree.
        """
        b = self.action_space_size
        L = self.num_layers
        first_of_last = (b ** L - 1) // (b - 1)
        return first_of_last <= idx


    def get_layers_indices(self, layer_number):
        """
        Get the indices of the layers in the tree.
        """
        b = self.action_space_size
        start_idx = (b ** layer_number - 1) // (b - 1)
        end_idx = (b ** (layer_number + 1) - 1) // (b - 1)

        return start_idx, end_idx


    def expand(self, all_scalars, root_hidden_state=None):
        self.value_evals, self.rewards, self.all_policy_probs = all_scalars
        if self.is_root:
            # set the root hidden state
            self.hidden_state = root_hidden_state
        else:
            # prepend the parent node value evaluation, reward and policy probs
            self.value_evals = th.cat((self.parent.get_value_eval().unsqueeze(0), self.value_evals), dim=0)
            self.rewards = th.cat((self.parent.get_reward().unsqueeze(0), self.rewards), dim=0)
            self.all_policy_probs = th.cat((self.parent.get_prior().unsqueeze(0), self.all_policy_probs), dim=0)

        self.priors = self.all_policy_probs # todo reconsider this


    def set_pll_args(self):
        self.seqs, positions, action_seq = self.bfs_sequences()
        self.positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        self.action_seq = torch.tensor(action_seq, dtype=torch.long, device=self.device)


    def get_seqs(self):
        if self.seqs is None:
            self.set_pll_args()
        return self.seqs

    def get_positions(self):
        if self.positions is None:
            self.set_pll_args()
        return self.positions

    def get_action_seq(self):
        if self.action_seq is None:
            self.set_pll_args()
        return self.action_seq


    def make_subtree_mask(self):
        device = self.device

        seqs = self.get_seqs()
        n = len(seqs)

        lengths = [len(seq) for seq in seqs]
        allowed_mask = [
            [lengths[j] <= lengths[i] and seqs[i][:lengths[j]] == seqs[j]
             for j in range(n)]
            for i in range(n)
        ]
        allowed_mask = torch.tensor(allowed_mask, dtype=torch.bool, device=device)
        # 2) Enforce causal ordering (tokens can only attend to earlier-or-same tokens)
        causal_order = torch.tril(torch.ones((n, n), dtype=torch.bool, device=device))
        mask = allowed_mask & causal_order
        return mask


    def get_subtree_mask(self):

        if self.mask is None:
        # Get the size of the subtree
            self.mask = self.make_subtree_mask()

        return self.mask





    def bfs_sequences(self):
        """
        Build every sequence you can make from `actions`, by increasing length
        up to `max_length`. Returns two lists:
          - sequences: List of the sequences (each a List[T])
          - lengths:   List of their lengths (int), same order as `sequences`
        """
        sequences = []
        lengths = []
        last_items = []

        for length in range(1, self.num_layers + 1):
            # product(self.actions, repeat=length) goes in lex order
            for seq in itertools.product(self.action_space, repeat=length):
                sequences.append(list(seq))
                lengths.append(length)
                last_items.append(seq[-1])

        return sequences, lengths, last_items



    def calc_entire_policy_value_and_variance_subtree(self):

        # reverse range
        for layer_num in reversed(range(0, self.num_layers+1)):
            layer = SubTreeLayer(self, layer_num)
            val, var = policy_value_and_variance_layer(layer, self.discount)
            layer.set_val_and_vars(val, var)

        return self.vals



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

# self.size, self.non_leaf_size = parent_tree.size, parent_tree.non_leaf_size # minus one if the parent is root
#             self.config, self.device, self.policy = parent_tree.config, parent_tree.device, parent_tree.policy
#             self.discount, self.action_space, self.action_space_size, self.num_layers = parent_tree.discount, parent_tree.action_space, parent_tree.action_space_size, parent_tree.num_layers
#
#             self.seqs, self.positions, self.action_seq, self.mask = parent_tree.seqs, parent_tree.positions, parent_tree.action_seq, parent_tree.mask
#             self.leaf_vars, self.leaf_inv_vars, self.leaf_vals = parent_tree.leaf_vars, parent_tree.leaf_inv_vars, parent_tree.leaf_vals
#             self.hidden_state = parent_tree.hidden_state
#
#             self.value_eval_vars, self.reward_vars = parent_tree.value_eval_vars, parent_tree.reward_vars
