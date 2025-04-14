import math

import numpy
import torch

from trans_zero.utils import models
from trans_zero.mvc_utils.policies import MeanVarianceConstraintPolicy
from trans_zero.mvc_utils.utility_functions import policy_value, compute_inverse_q_variance, get_children_inverse_variances
from .node import Node, MVCNode

from abc import ABC, abstractmethod
import time
from collections import defaultdict


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

        # todo, could be cleaned up
        if self.config.PUCT_variant == "visit":
            self.puct = PUCT_visit(config)
            self.unexpanded_root = Node(0, self.config)
        elif self.config.PUCT_variant == "mvc":
            self.puct = PUCT_MVC(config)
            self.unexpanded_root = MVCNode(0, self.config) # todo, stop sending the config around, keep it central
        else:
            raise NotImplementedError(f"PUCT variant {self.config.PUCT_variant} not implemented.")

        if self.config.test_ucb: # todo clean
            timestamp = time.time()
            self.file_path = f"data/test_scores/test_ucb_scores_{timestamp}.csv"

            self.min_max_stats_std = MinMaxStats()
            with open(self.file_path, "w") as f:
                #f.write("P,C,UCB,UCB_mvc,U,U_mvc,Q,Q_mvc\n")
                f.write("P_var,C_var,P_vis,C_vis\n")
                f.flush


    def expand_root(self, root, observation, model, legal_actions, to_play):
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
        )

        # Initial step
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)

        # Make the root predicted value and reward a scalar
        root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
        reward = models.support_to_scalar(reward, self.config.support_size).item()

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root.expand(legal_actions, to_play, root_predicted_value, reward, policy_logits, hidden_state)

        return root, root_predicted_value


    def init_root(self, observation, model, legal_actions, to_play, override_root_with, add_exploration_noise):
        # print(f"is trans net: {is_trans_net}")
        if override_root_with:
            expanded_root = override_root_with
            root_predicted_value = None
        else:
            expanded_root, root_predicted_value = self.expand_root(self.unexpanded_root, observation, model, legal_actions, to_play)

        if add_exploration_noise:
            expanded_root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        return expanded_root, root_predicted_value


    def selection(self, node, min_max_stats):
        actions = []
        virtual_to_play = node.to_play
        current_tree_depth = 0

        while node.expanded():
            current_tree_depth += 1
            action, node = self.select_child(node, min_max_stats)  # tree action selection
            actions.append(action)

            # Players play turn by turn
            if virtual_to_play + 1 < len(self.config.players):
                virtual_to_play = self.config.players[virtual_to_play + 1]
            else:
                virtual_to_play = self.config.players[0]

        return node, actions, current_tree_depth, virtual_to_play


    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        transformer_net = self.config.network == "transformer"

        root, root_predicted_value = self.init_root(observation, model, legal_actions, to_play, override_root_with, add_exploration_noise)

        min_max_stats = MinMaxStats()
        max_tree_depth = 0

        for _ in range(self.config.num_simulations):

            (
                node,
                actions,
                current_tree_depth,
                virtual_to_play,
            ) = self.selection(root, min_max_stats)


            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            if transformer_net:
                latent_root_state = root.hidden_state

                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    None,
                    None,
                    latent_root_state = latent_root_state,
                    action_sequence= torch.tensor([actions]).to(latent_root_state.device),
                )

            else:
                parent = node.parent
                action = actions[-1]
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[action]]).to(parent.hidden_state.device),
                )

            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()

            node.expand(
                self.config.action_space,
                virtual_to_play,
                value,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(node, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }

        return root, extra_info



    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score without recalculating.
        """
        # Calculate scores once, store action-score pairs
        # if self.config.PUCT_U == "mvc":
        #     node.reset_var()

        action_score_pairs = [
            (action, self.puct.ucb_score(node, child, min_max_stats))
            for action, child in node.children.items()
        ]

        # if self.config.test_ucb:
        #     res_U = [self.U_comparator(node, child) for action, child in node.children.items()]
        #     self.write_test2_to_file(res_U)

            #self.write_test_to_file(node, min_max_stats)

        # Find the maximum UCB score
        max_ucb = max(score for action, score in action_score_pairs)

        # Select actions that have the maximum UCB score
        best_actions = [action for action, score in action_score_pairs if score == max_ucb]

        # Randomly select among best actions
        selected_action = numpy.random.choice(best_actions)

        return selected_action, node.children[selected_action]


    def backpropagate(self, node, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root
        """
        # todo, try new backprop
        if len(self.config.players) == 1:
            while True:
                node.value_sum += value
                node.increment_visit_count()
                value = node.reward + self.config.discount * value
                min_max_stats.update(value)

                # resetting in the backprop, NEW, todo remove other
                node.variance = None
                node.policy_value = None

                if node.parent is None:
                    break
                node = node.parent


        elif len(self.config.players) == 2:
            while True:
                node.value_sum += value if node.to_play == to_play else -value
                node.increment_visit_count()
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

                if node.parent is None:
                    break
                node = node.parent

        else:
            raise NotImplementedError("More than two player mode not implemented.")




    # def U_comparator(self, parent, child):
    #     par_inv_q_var = compute_inverse_q_variance(parent, self.policy, self.config.discount)
    #     child_inv_q_var = compute_inverse_q_variance(child, self.policy, self.config.discount)
    #
    #     par_visit = parent.visit_count
    #     child_visit = child.visit_count
    #     return par_inv_q_var, child_inv_q_var, par_visit, child_visit

    # todo consider causal mask in forward pass of transformer

    # def ucb_score_test(self, parent, child, min_max_stats):
    #     u_std, u_mvc = self.calc_U_std(parent, child), self.calc_U_mvc(parent, child)
    #     q_std, q_mvc = self.calc_Q_std(child, self.min_max_stats_std), self.calc_Q_mvc(child, min_max_stats)
    #     return u_std, u_mvc, q_std, q_mvc


    # todo reconsider what to do with these
    def write_test_to_file(self, node, min_max_stats):
        test_scores = [
            (node.name, child.name, self.ucb_score_test(node, child, min_max_stats))
            for action, child in node.children.items()
        ]
        # append to file
        with open(self.file_path, "a") as f:
            for test_score in test_scores:
                # round all test scores to 2 decimals
                p_name, c_name, scores = test_score
                # round the scores to 2 decimals, its tensors
                # turn all tensors to floats, leave all floats as they are
                scores = [score.item() if hasattr(score, "item") else score for score in scores]

                scores = [round(score, 2) for score in scores]
                U, U_mvc, Q, Q_mvc = scores
                UCB = Q + U
                UCB_mvc = Q_mvc + U_mvc

                f.write(f"{p_name},{c_name},{UCB},{UCB_mvc},{U},{U_mvc},{Q},{Q_mvc}\n")
            f.flush()

    def write_test2_to_file(self, test_scores):
        # append to file
        with open(self.file_path, "a") as f:
            for test_score in test_scores:
                # round all test scores to 2 decimals
                par_inv_q_var, child_inv_q_var, par_visit, child_visit = test_score
                f.write(f"{par_inv_q_var},{child_inv_q_var},{par_visit},{child_visit}\n")
            f.flush()


class MCTS_PLL_1(MCTS):
    def __init__(self, config):
        super().__init__(config)

    def run(
            self,
            model,
            observation,
            legal_actions,
            to_play,
            add_exploration_noise,
            override_root_with=None,
    ):

        # print(f"is trans net: {is_trans_net}")
        root, root_predicted_value = self.init_root(observation, model, legal_actions, to_play, override_root_with, add_exploration_noise)

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        latent_root_state = root.hidden_state
        latent_root_state = latent_root_state.repeat(self.config.expansion_budget, 1)  # todo dirty trick, fix better later

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            actions = []
            current_tree_depth = 0

            # todo reconsider where this should be
            #node.reset_var()

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)  # tree action selection
                search_path.append(node)
                actions.append(action)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            action_sequences_from_node = self.bfs_actions(self.config.expansion_budget, legal_actions)
            # prepend actions to each
            action_sequences = [actions + action_sequence for action_sequence in action_sequences_from_node]
            # make into torch tensor

            padded_as, pad_masks = self.pad_action_sequences(action_sequences)
            padded_as = torch.tensor(padded_as).to(latent_root_state.device)
            # # add singleton dimension as last
            pad_masks = torch.tensor(pad_masks).to(latent_root_state.device)

            # copy root hidden state into size of action sequences in dim 0

            all_values, all_rewards, all_policy_logits, _ = model.recurrent_inference_fast(
                latent_root_state=latent_root_state,
                action_sequence=padded_as,
                mask=pad_masks,
                use_causal_mask = True
            )

            org_node = node
            for i, action_sequence in enumerate(action_sequences_from_node):
                for action in action_sequence:
                    node = node.children[action]

                #full_ac_seq = actions + action_sequence
                # todo try recurrent inf fast
                # value_i, reward_i, policy_logits_i, hidden_state = model.recurrent_inference(
                #     None, None,
                #     latent_root_state = latent_root_state,
                #     action_sequence=torch.tensor(full_ac_seq).unsqueeze(0).to(latent_root_state.device),
                # )

                last_action_idx = len(actions) + len(action_sequence) # not -1 since the observation takes one token (todo see over implementation when using larger token space for obs)

                # just take the last ones
                # ":" is to keep dimension
                value_i = all_values[i][[last_action_idx]]
                reward_i = all_rewards[i][[last_action_idx]]
                policy_logits_i = all_policy_logits[i][[last_action_idx]]

                # todo emil idea check unceratnity based on several values?
                # todo alternative emil, instead of expanding bottom up, expand all at once.
                # for example if having a budget of 6 you could expand A, B, C, AA, AB, AC, or instead do
                # AA, BA, CA, AB, BB, CB and just fill in the values for A, B and C (given they are already predicted)

                # todo also try longer runs and not capping after terminal state
                value_i = models.support_to_scalar(value_i, self.config.support_size).item()
                reward_i = models.support_to_scalar(reward_i, self.config.support_size).item()

                node.expand(
                    legal_actions,
                    virtual_to_play,
                    value_i,
                    reward_i,
                    policy_logits_i,
                    latent_root_state, # todo, make optional
                )

                self.backpropagate(node, value_i, virtual_to_play, min_max_stats)

                node = org_node

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }

        return root, extra_info


    @staticmethod
    def bfs_actions(budget, action_space):
        """
        Generate sequences of actions in a breadth-first search manner, starting from an empty sequence.

        Parameters:
            budget (int): Total number of sequences to generate.
            action_space (list): List of actions available.

        Returns:
            list: A list of lists, where each inner list is a sequence of actions.
        """
        from collections import deque

        results = []
        queue = deque([[]])  # Start with the empty sequence

        while queue and len(results) < budget:
            sequence = queue.popleft()
            results.append(sequence)
            if len(results) >= budget:
                break
            # Generate children by appending each possible action
            for action in action_space:
                new_sequence = sequence + [action]
                queue.append(new_sequence)

        return results

    @staticmethod
    def pad_action_sequences(action_sequences, pad_value=0):
        # Get the length of the longest sequence, assumed to be the last one.
        max_length = len(action_sequences[-1])

        padded_sequences = []
        pad_masks = []

        for seq in action_sequences:
            # Calculate the number of pad tokens needed for the current sequence.
            padding_needed = max_length - len(seq)

            # Pad the sequence.
            padded_seq = seq + [pad_value] * padding_needed
            padded_sequences.append(padded_seq)

            # Create a mask: False for original entries, True for padded ones.
            # todo the + 1 is for the state, should be more if state is more tokens
            mask = [False] * (len(seq) + 1) + [True] * padding_needed
            pad_masks.append(mask)

        return padded_sequences, pad_masks


class MCTS_PLL_2(MCTS_PLL_1):
    def __init__(self, config):
        super().__init__(config)


    def get_unique_nodes_for_sequence(self, sequences):
        unique_occurrences = {}

        # Process each input sequence (only one pass over each)
        for seq in sequences:
            history = []  # will collect the history (as lowercase tokens)
            for token in seq:
                pos = len(history) + 1  # positions start at 1
                # Construct full_seq as (history plus current token in lowercase)
                full_seq = tuple(history) + (token,)
                key = (token, pos, full_seq)
                unique_occurrences[key] = key
                history.append(token)

        return unique_occurrences

    def make_causal_mask(self, seqs, device):
        n = len(seqs)
        # Pre-cache the full sequences and their lengths.
        lengths = [len(seq) for seq in seqs]

        # Compute the allowed (visible) mask based on prefix logic.
        allowed_mask = [
            [lengths[j] <= lengths[i] and seqs[i][:lengths[j]] == seqs[j]
             for j in range(n)]
            for i in range(n)
        ]
        allowed_mask_tensor = torch.tensor(allowed_mask, dtype=torch.bool, device=device)

        # Create a new allowed mask for (n+1)x(n+1).
        # We initialize with True, meaning "allowed".
        new_allowed_mask = torch.ones((n + 1, n + 1), dtype=torch.bool, device=device)
        # Insert the computed mask into the lower-right block.
        new_allowed_mask[1:, 1:] = allowed_mask_tensor

        # Invert the mask.
        # Now False indicates allowed (visible) and True indicates a masked-out position.
        final_mask = ~new_allowed_mask
        print(final_mask)
        return final_mask

    def unwind_sequences(self, sequences):
        """
        For a list of sequences of actions (each a list of action labels),
        this function computes for each (node, position) the set of unique histories (prior nodes),
        and then "unwinds" them into a single action sequence and a corresponding positional encoding.

        The final encoding ensures positions are non-decreasing.
        """
        groups = {}  # (node, pos) -> set of histories (each history is a tuple)
        order = {}  # (node, pos) -> first occurrence order (to preserve encounter order)
        next_order = 0

        # Loop once over each sequence to update groups
        for seq in sequences:
            history = []
            for pos, node in enumerate(seq, start=1):
                key = (node, pos)
                if key not in groups:
                    groups[key] = set()
                    order[key] = next_order
                    next_order += 1
                groups[key].add(tuple(history))
                history.append(node)

        # Now sort keys by position (non-decreasing) and then by encounter order.
        sorted_keys = sorted(groups.keys(), key=lambda k: (k[1], order[k]))

        action_seq = []
        pos_seq = []
        for node, pos in sorted_keys:
            count = len(groups[(node, pos)])
            action_seq.append(node * count)  # repeat node as many times as there are unique histories
            pos_seq.append([pos] * count)

        return action_seq, pos_seq


    def get_action_sequence(self, actions, latent_root_state, legal_actions):
        action_sequences_from_node = self.bfs_actions(self.config.expansion_budget, legal_actions)
        # prepend actions to each
        action_sequences = [actions + action_sequence for action_sequence in action_sequences_from_node]
        # get unique nodes for each sequence
        # action_seq, pos_indices = self.unwind_sequences(action_sequences)
        unique_occurrences = self.get_unique_nodes_for_sequence(action_sequences)

        tokens_list = list(unique_occurrences.keys())
        # Sort tokens by: position, then by token label (alphabetically), then by full_seq.
        tokens_list.sort(key=lambda x: (x[1], x[0], x[2]))

        # Build the action sequence and positional sequence.
        action_seq, pos_indices, full_seqs = [], [], []

        for token, pos, full_seq in tokens_list:
            action_seq.append(token)
            pos_indices.append(pos)
            full_seqs.append(list(full_seq))

        # Now build the causal mask.
        causal_mask = self.make_causal_mask(full_seqs, device=latent_root_state.device)
        # unsqueeze last and first dim
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(-1)

        # make into torch tensor
        action_seq = torch.tensor(action_seq).to(latent_root_state.device).unsqueeze(0)
        pos_indices = torch.tensor(pos_indices).to(latent_root_state.device)

        return action_seq, pos_indices, full_seqs, causal_mask


    def run(
            self,
            model,
            observation,
            legal_actions,
            to_play,
            add_exploration_noise,
            override_root_with=None,
    ):

        # print(f"is trans net: {is_trans_net}")
        root, root_predicted_value = self.init_root(observation, model, legal_actions, to_play, override_root_with, add_exploration_noise)

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        latent_root_state = root.hidden_state
        #latent_root_state = latent_root_state.repeat(self.config.expansion_budget, 1)  # todo dirty trick, fix better later

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            actions = []
            current_tree_depth = 0

            # todo reconsider where this should be
            #node.reset_var()

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)  # tree action selection
                actions.append(action)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            action_seq, pos_indices, full_seqs, causal_mask = self.get_action_sequence(actions, latent_root_state, legal_actions)

            # copy root hidden state into size of action sequences in dim 0

            all_values, all_rewards, all_policy_logits, _ = model.recurrent_inference(
                None,None,
                latent_root_state=latent_root_state,
                action_sequence=action_seq,
                custom_pos_indices = pos_indices,
                custom_causal_mask=causal_mask,
                return_only_last_prediction=False
            )

            org_node = node
            for i, action_sequence in enumerate(full_seqs):
                for action in action_sequence[1:]:
                    node = node.children[action]

                last_action_idx = len(action_sequence) # not -1 since the observation takes one token (todo see over implementation when using larger token space for obs)

                value_i = all_values[:, last_action_idx, :]
                reward_i = all_rewards[:, last_action_idx, :]
                policy_logits_i = all_policy_logits[:, last_action_idx, :]

                # todo emil idea check unceratnity based on several values?
                # todo alternative emil, instead of expanding bottom up, expand all at once.
                # for example if having a budget of 6 you could expand A, B, C, AA, AB, AC, or instead do
                # AA, BA, CA, AB, BB, CB and just fill in the values for A, B and C (given they are already predicted)

                # todo also try longer runs and not capping after terminal state
                assert ((value_i >= -self.config.support_size) & (value_i <= self.config.support_size)).all(), f"value_i out of bounds: {value_i}"

                value_i = models.support_to_scalar(value_i, self.config.support_size).item()
                reward_i = models.support_to_scalar(reward_i, self.config.support_size).item()

                node.expand(
                    legal_actions,
                    virtual_to_play,
                    value_i,
                    reward_i,
                    policy_logits_i,
                    latent_root_state, # todo, make optional
                )

                self.backpropagate(node, value_i, virtual_to_play, min_max_stats)
                node = org_node

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }

        return root, extra_info




class PUCT(ABC):
    """
    PUCT (Predictor + UCT) is a Monte Carlo Tree Search algorithm that uses a
    predictor to guide the search.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def U(self, parent, child):
        """
        Compute the exploration bonus for a child node.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def Q(self, child):
        """
        Compute the value estimate for a child node.
        Must be implemented by subclasses.
        """
        pass

    def ucb_score(self, parent, child, min_max_stats):
        """
        Compute the PUCT (Predictor + UCT) score for a child node.
        Score = Q + U
        """
        U = child.prior * self.U(parent, child)
        Q = self.Q(child)
        Q = min_max_stats.normalize(Q)
        return Q + U


class PUCT_visit(PUCT):
    """
    PUCT (Predictor + UCT) is a Monte Carlo Tree Search algorithm that uses a
    predictor to guide the search.
    """

    def __init__(self, config):
        super().__init__(config)


    def U(self, parent, child):
        pb_c_log = math.log(
            (parent.get_visit_count() + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c = pb_c_log + self.config.pb_c_init
        pb_c *= math.sqrt(parent.get_visit_count()) / (child.get_visit_count() + 1)
        return pb_c


    def Q(self, child):
        if child.get_visit_count() < 1:
            return 0

        raw_value = child.reward + self.config.discount * (
            child.value() if len(self.config.players) == 1 else -child.value()
        )
        return raw_value


class PUCT_MVC(PUCT):
    def __init__(self, config):
        super().__init__(config)
        self.policy = MeanVarianceConstraintPolicy(config=self.config)


    def Q(self, child):
        return policy_value(child, self.policy, self.config.discount)

    def U(self, parent, child):
        par_inv_q_var = compute_inverse_q_variance(parent, self.policy, self.config.discount)
        child_inv_q_var = compute_inverse_q_variance(child, self.policy, self.config.discount)

        return self.config.PUCT_C * (math.sqrt(par_inv_q_var) / (child_inv_q_var + 1))

    # todo try, seems not to work
    def calc_U_mvc_experimental(self, parent, child):
        par_inv_q_var = compute_inverse_q_variance(parent, self.policy, self.config.discount)
        child_inv_q_var = compute_inverse_q_variance(child, self.policy, self.config.discount)

        par_inv_q_var = max(par_inv_q_var/3 - 1, 0)
        child_inv_q_var = max(child_inv_q_var/3 - 1, 0)

        pb_c_log = math.log(
            (par_inv_q_var + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c = pb_c_log + self.config.pb_c_init
        pb_c *= math.sqrt(par_inv_q_var) / (child_inv_q_var + 1)
        return pb_c


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

