

import math

import numpy
import torch

from trans_zero.utils import models
from .node import Node, MVCNode, SubTree, SubTreeNode

from abc import ABC, abstractmethod
import time

from ..utils.other_utils import arg_max_with_tie_breaking


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
        self.min_max_stats = MinMaxStats()


        # todo, could be cleaned up
        if self.config.PUCT_variant == "mvc" or self.config.action_selection == "mvc" or self.config.policy_target_type == "mvc":
            # self.puct = PUCT_MVC(config)
            self.unexpanded_root = MVCNode(0, self.config)  # todo, stop sending the config around, keep it central
        else:
            #self.puct = PUCT_visit(config)
            self.unexpanded_root = Node(0, self.config)

        if self.config.PUCT_variant == "mvc":
            self.puct = PUCT_MVC(config)
        else:
            self.puct = PUCT_visit(config)


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
        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in legal_actions]), dim=0
        ).tolist()

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root.expand(legal_actions, to_play, root_predicted_value, reward, policy_values, hidden_state)

        return root, root_predicted_value


    def init_root(self, observation, model, legal_actions, to_play, override_root_with, add_exploration_noise, pll_args = None):
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


    def selection(self, node):
        actions = []
        virtual_to_play = node.to_play
        current_tree_depth = 0

        while node.expanded():
            current_tree_depth += 1
            action, node = self.select_child(node)  # tree action selection
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
        torch.cuda.reset_peak_memory_stats()
        transformer_net = self.config.network == "transformer"

        root, root_predicted_value = self.init_root(observation, model, legal_actions, to_play, override_root_with, add_exploration_noise)
        # important! if a node has just been expanded this is needed
        root.recalculate_val_and_var(newly_expanded=True)

        max_tree_depth = 0

        for _ in range(self.config.num_simulations):

            (
                node,
                actions,
                current_tree_depth,
                virtual_to_play,
            ) = self.selection(root)


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
            policy_values = torch.softmax(
                torch.tensor([policy_logits[0][a] for a in self.config.action_space]), dim=0
            ).tolist()

            node.expand(
                self.config.action_space,
                virtual_to_play,
                value,
                reward,
                policy_values,
                hidden_state,
            )

            self.backpropagate(node, value, virtual_to_play)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }



        return root, extra_info



    def select_child(self, node):
        """
        Select the child with the highest UCB score without recalculating.
        """
        # Calculate scores once, store action-score pairs
        # if self.config.PUCT_U == "mvc":
        #     node.reset_var()

        ucb_scores = self.puct.get_ucb_scores(node, self.min_max_stats)

        # if self.config.test_ucb:
        #     res_U = [self.U_comparator(node, child) for action, child in node.children.items()]
        #     self.write_test2_to_file(res_U)

            #self.write_test_to_file(node, min_max_stats)

        # Find the maximum UCB score
        max_ucb = max(ucb_scores)

        # Select actions that have the maximum UCB score
        best_actions = [action for action, score in zip(node.action_space, ucb_scores) if score == max_ucb]

        # Randomly select among best actions
        selected_action = numpy.random.choice(best_actions)

        return selected_action, node.get_child(selected_action)


    def backpropagate(self, node, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root
        """
        newly_expanded=True
        if len(self.config.players) == 1:
            while True:
                if not isinstance(node, MVCNode):
                    node.value_sum += value
                    node.increment_visit_count()
                    value = node.reward + self.config.discount * value
                    self.min_max_stats.update(node.reward + self.config.discount * node.get_value())

                else:
                    node.recalculate_val_and_var(newly_expanded=newly_expanded)
                    newly_expanded = False
                    self.min_max_stats.update(node.get_value())

                if node.parent is None:
                    break

                node.parent.set_children_val_and_vars(node)
                node = node.parent



        elif len(self.config.players) == 2:
            while True:
                node.value_sum += value if node.to_play == to_play else -value
                node.increment_visit_count()
                self.min_max_stats.update(node.reward + self.config.discount * -node.get_value())

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

    # def ucb_score_test(self, parent, child, min_max_stats):
    #     u_std, u_mvc = self.calc_U_std(parent, child), self.calc_U_mvc(parent, child)
    #     q_std, q_mvc = self.calc_Q_std(child, self.min_max_stats_std), self.calc_Q_mvc(child, min_max_stats)
    #     return u_std, u_mvc, q_std, q_mvc


    # todo reconsider what to do with these
    def write_test_to_file(self, node):
        test_scores = [
            (node.name, child.name, self.ucb_score_test(node, child, self.min_max_stats))
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



class MCTS_PLL(MCTS):
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
        device = next(model.parameters()).device
        legal_actions_tensor = torch.tensor(legal_actions, device=device)

        action_seq_for_pred, pos_indices, full_seqs = self.get_action_sequence([], device, legal_actions, budget=self.config.expansion_budget+1, include_root=True)
        pll_args = {
            "action_sequence": action_seq_for_pred,
            "custom_pos_indices": pos_indices,
            "full_seqs": full_seqs,

            "return_n_last_predictions": action_seq_for_pred.size(1) + 1,
            "custom_causal_mask": self.make_causal_mask(full_seqs, device=device),
        }

        root, all_scalars = self.expand_root_pll(self.unexpanded_root, observation, model, legal_actions, to_play, pll_args)

        if False and add_exploration_noise: # TODO
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        max_tree_depth = 0

        latent_root_state = root.hidden_state
        #latent_root_state = latent_root_state.repeat(self.config.expansion_budget, 1)  # todo dirty trick, fix better later

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            actions = []
            current_tree_depth = 0

            # log n
            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node)  # tree action selection
                actions.append(action)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # expand all children instead of just ucb highest one
            if self.config.expand_all_children:
                actions = actions[:-1]
                node = node.parent

            action_seq_for_pred, pos_indices, full_seqs = self.get_action_sequence(actions, device, legal_actions, budget=self.config.expansion_budget, include_root = not self.config.expand_all_children)
            causal_mask = self.make_causal_mask(full_seqs, device=device)


            all_values, all_rewards, all_policy_logits, _ = model.recurrent_inference(
                None, None,
                latent_root_state=latent_root_state,
                action_sequence=action_seq_for_pred,
                custom_pos_indices = pos_indices,
                custom_causal_mask=causal_mask,
                return_n_last_predictions=self.config.expansion_budget
                )

            all_scalars = self.get_scalars(all_values, all_rewards, all_policy_logits, legal_actions_tensor)

            all_actions_from_node = self.trim_sequences(full_seqs, len(actions))
            all_actions_from_node = all_actions_from_node[-self.config.expansion_budget:]

            self.multi_expansion(all_scalars, node, all_actions_from_node, legal_actions, virtual_to_play,latent_root_state)
            max_tree_depth = max(max_tree_depth, current_tree_depth)


        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": None # todo root_predicted_value,
        }

        return root, extra_info

    def bfs_actions(self, budget, action_space, include_root = False):
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
        queue = deque([[]]) if include_root else deque([[a] for a in action_space])  # Start with the empty sequence

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


    def get_unique_nodes_for_sequence(self, sequences):
        unique_occurrences = {}

        # Process each input sequence (only one pass over each)
        for seq in sequences:
            history = []  # will collect the history (as lowercase tokens)
            for token in seq:
                pos = len(history)  # positions start at 1
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

        # Compute the allowed mask for the tokens based on the prefix condition.
        # For each token i and j in tokens_list, allowed if token i's full_seq starts with token j's full_seq.
        allowed_mask = [
            [lengths[j] <= lengths[i] and seqs[i][:lengths[j]] == seqs[j]
             for j in range(n)]
            for i in range(n)
        ]
        allowed_mask_tensor = torch.tensor(allowed_mask, dtype=torch.bool, device=device)

        # Enforce causal ordering (i.e. j <= i) to create a normal causal mask.
        # For tokens positions (indices 0 to n-1 corresponding to tokens_list), we construct a lower-triangular mask.
        causal_order = torch.tril(torch.ones((n, n), dtype=torch.bool, device=device))
        final_allowed_tokens = allowed_mask_tensor & causal_order

        # Create a new mask that is (n+1) x (n+1) to include the global token at index 0.
        new_allowed = torch.zeros((n + 1, n + 1), dtype=torch.bool, device=device)

        # Global token row (index 0): allow only itself.
        new_allowed[0, 0] = True

        # For all token rows (indices 1 to n), always allow attending to the global token (column 0).
        new_allowed[1:, 0] = True

        # For the remaining positions, use the computed allowed tokens mask.
        new_allowed[1:, 1:] = final_allowed_tokens

        # Convert boolean mask to the float mask expected by torch.nn,
        # where allowed positions become 0 and masked out positions become -inf.
        attention_mask = torch.where(new_allowed, torch.tensor(0.0), torch.tensor(float('-inf')))
        return attention_mask



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


    def get_action_sequence(self, actions_from_star_node, device, legal_actions, budget, include_root=False):
        subtree_actions = self.bfs_actions(budget, legal_actions, include_root)
        # prepend actions to each
        action_sequences = [actions_from_star_node + action_sequence for action_sequence in subtree_actions]
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


        # unsqueeze last and first dim

        # make into torch tensor
        action_seq = torch.tensor(action_seq).to(device).unsqueeze(0)
        pos_indices = torch.tensor(pos_indices).to(device)
        return action_seq, pos_indices, full_seqs

    @staticmethod
    def trim_sequences(sequences, action_length):
        return [seq[action_length:] for seq in sequences if len(seq) >= action_length]


    def expand_root_pll(self, root, observation, model, legal_actions, to_play, pll_args):
        device = next(model.parameters()).device
        legal_actions_tensor = torch.tensor(legal_actions, device=device)

        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(device)
        )

        full_seqs = pll_args.pop("full_seqs")
        # Initial step
        start_time = time.time()
        (
            all_values,
            all_rewards,
            all_policy_logits,
            root_latent_state,
        ) = model.initial_inference(observation, pll_args=pll_args)

        all_scalars = self.get_scalars(all_values, all_rewards,all_policy_logits, legal_actions_tensor)

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        #root.expand(legal_actions, to_play, value, reward, policy_values,)
        full_seqs = [[]] + full_seqs
        self.multi_expansion(all_scalars, root, full_seqs, legal_actions, to_play, root_latent_state)
        multi_exp_time = time.time()

        #print(f"Init inf time: {init_inf_time - start_time:.4f}s, scalar time: {scalar_time - init_inf_time:.4f}s, multi_exp time: {multi_exp_time - scalar_time:.4f}s")
        return root, all_scalars


    def get_scalars(self, all_values, all_rewards, all_policy_logits, legal_actions_tensor):
        return (models.support_to_scalar(all_values, self.config.support_size),
                models.support_to_scalar(all_rewards, self.config.support_size),
                torch.softmax(all_policy_logits[:, legal_actions_tensor], dim=-1))


    def multi_expansion(self, all_scalars, node, all_actions_from_node, legal_actions, virtual_to_play,
                        root_latent_state):
        all_value_scalars, all_reward_scalars, all_policy_probs = all_scalars
        org_node = node

        node_list = []  # todo turn into queue

        while node.parent is not None:
            node = node.parent
            node_list.append(node)

        # reverse list
        node_list.reverse()
        node = org_node

        for i, actions_from_node in enumerate(all_actions_from_node):

            for action in actions_from_node:
                node = node.children[action]

            last_action_idx = i

            value_i = all_value_scalars[last_action_idx].item()
            reward_i = all_reward_scalars[last_action_idx].item()
            policy_values_i = all_policy_probs[last_action_idx].tolist()

            # todo also try longer runs and not capping after terminal state
            node.expand(
                legal_actions,
                virtual_to_play,
                value_i,
                reward_i,
                policy_values_i,
                root_latent_state,  # todo, make optional
            )

            node_list.append(node)
            node = org_node

        self.backpropagate_more(node_list)

        #print(f"List time: {list_time - start_time:.4f}s, for loop time: {for_loop_time - list_time:.4f}s, backprop time: {backprop_time - for_loop_time:.4f}s")

    def backpropagate_more(self, node_list):

        # reverse traversal to get children first
        val_list = []
        for node in reversed(node_list):
            val, _ = node.recalculate_val_and_var()
            val_list.append(val.item())

            if node.parent is None:
                break

            node.parent.set_children_val_and_vars(node)

        self.min_max_stats.mass_update(val_list)


class MCTS_SubTree(MCTS_PLL):
    def __init__(self, config, device, override_root_with=None):
        super().__init__(config)
        self.device = device
        if override_root_with is not None:
            override_root_with.reinit()
            self.unexpanded_subtree_root = override_root_with
        else:
            self.unexpanded_subtree_root = SubTree(None, config, device)
        self.puct = PUCT_Subtree(config, device)


    def run(
            self,
            model,
            observation,
            legal_actions,
            to_play,
            add_exploration_noise,
            override_root_with=None,
    ):

        legal_actions_tensor = torch.tensor(legal_actions, device=self.device)

        pll_args = self.get_pll_args(self.unexpanded_subtree_root, [])

        root_subtree = self.expand_subtree_root(self.unexpanded_subtree_root, observation, model, legal_actions, to_play, pll_args)

        if add_exploration_noise:
            root_subtree.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        vals = root_subtree.calc_entire_policy_value_and_variance_subtree()


        self.min_max_stats.mass_update_tensor(vals)

        max_tree_depth = 0

        latent_root_state = root_subtree.hidden_state
        #latent_root_state = latent_root_state.repeat(self.config.expansion_budget, 1)  # todo dirty trick, fix better later

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            subtree = root_subtree
            node_idx = 0

            actions = []
            current_tree_depth = 0

              # tree action selection
            # log n


            node = SubTreeNode(subtree, node_idx)

            while node.expanded(): # todo could batch calc UCB scores, but dont know if faster
                current_tree_depth += 1
                action, node = self.select_child(node)  # tree action selection
                actions.append(action) # temp

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            unexpanded_subtree = SubTree(connecting_node=node)
            pll_args = self.get_pll_args(unexpanded_subtree, actions)

            # start cuda time


            all_values, all_rewards, all_policy_logits, _ = model.recurrent_inference(
                None, None,
                latent_root_state=latent_root_state,
                **pll_args,
                )

            all_scalars = self.get_scalars(all_values, all_rewards, all_policy_logits, legal_actions_tensor)

            unexpanded_subtree.expand(all_scalars)
            self.backpropagate_subtree(unexpanded_subtree)

            #print(f"Expansion time: {expansion_time - start_time:.4f}s, backprop time: {backprop_time - expansion_time:.4f}s")

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": None # todo root_predicted_value,
        }

        root = SubTreeNode(root_subtree, 0)
        return root, extra_info


    def get_pll_args(self, unexpanded_tree, action_seq_to_node_star):

        action_seq_from_node_star = unexpanded_tree.get_action_seq()
        ac_seq_len = len(action_seq_to_node_star)
        action_seq_to_node_star = torch.tensor(action_seq_to_node_star, device=self.device, dtype=action_seq_from_node_star.dtype)
        return {
            "action_sequence": torch.cat((action_seq_to_node_star, action_seq_from_node_star), dim=0),
            "return_n_last_predictions": unexpanded_tree.size - 1 + unexpanded_tree.is_root,
            "custom_pos_indices": self.get_pos_indices(unexpanded_tree, prepend_len=ac_seq_len),
            "custom_causal_mask": self.make_causal_mask_subtree(self.unexpanded_subtree_root, num_global_tokens=1 + ac_seq_len),
        }


    def get_pos_indices(self, subtree, prepend_len):


        # shift the subtree positions by prepend_len
        shifted = subtree.get_positions() + prepend_len

        prefix = torch.arange(
            1,
            prepend_len + 1,
            dtype=shifted.dtype,
            device=self.device
        )
        # concat and return as a 1D tensor
        return torch.cat([prefix, shifted], dim=0)


    @staticmethod
    def build_sequence_tensor(n, seq):

        # create the [1..n] prefix on seq's device/dtype
        prefix = torch.arange(1, n + 1, device=seq.device, dtype=seq.dtype)

        # shift seq by n
        shifted = seq + n

        # concatenate and return
        return torch.cat([prefix, shifted], dim=0)



    def expand_subtree_root(self, root_tree, observation, model, legal_actions, to_play, pll_args):
        legal_actions_tensor = torch.tensor(legal_actions, device=self.device)

        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        # Initial step
        (
            all_values,
            all_rewards,
            all_policy_logits,
            root_latent_state,
        ) = model.initial_inference(observation, pll_args=pll_args)

        all_scalars = self.get_scalars(all_values, all_rewards, all_policy_logits, legal_actions_tensor)

        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root_tree.expand(all_scalars, root_latent_state)



        return root_tree


    def make_causal_mask_subtree(self, subtree, num_global_tokens=1):
        """
        Build a causal attention mask with `num_global_tokens` global positions.

        All tokens (real+global) can attend to any global token.
        Global tokens attend to each other only causally (i.e. triangular).
        Real tokens obey subtree.get_subtree_mask() causality.
        """
        # 1) get the n×n boolean mask for your real tokens
        real_mask = subtree.get_subtree_mask()  # True = allowed
        n = real_mask.size(0)
        G = num_global_tokens
        total = n + G

        # prepare constants
        device = self.device
        neg_inf = float('-inf')
        zero = torch.tensor(0.0, device=device)

        # 1) start fully masked
        mask = torch.full((total, total), neg_inf, device=device)

        # 2) real→real from subtree_mask
        #    where real_mask is True, set 0.0
        mask[G:, G:] = torch.where(real_mask, zero, neg_inf)

        if G > 0:
            # 3) global→global causal: allow [i,j] only if j <= i
            tri = torch.tril(torch.ones((G, G), dtype=torch.bool, device=device))
            mask[:G, :G][tri] = zero

            # 4) real→global: allow all real rows to see all globals
            mask[G:, :G] = zero

        return mask


    def backpropagate_subtree(self, subtree):
        # first calc entire subtree that the node is in
        vals = subtree.calc_entire_policy_value_and_variance_subtree()

        # TODO, if not root you can set the values of the parent to root of previous
        # todo update the parent of the root node to the values of the root node
        subtree.set_val_and_var_probs_connecting_node()

        # since root.parent is already set above, take the grandparent
        node = subtree.connecting_node.parent

        loop_vals = []
        while True:
            val, _ = node.recalculate_val_and_var()
            loop_vals.append(val)

            if node.parent is None:
                break

            if node.idx == 0: # otherwise this is skipped and not looked at later
                node.subtree.set_val_and_var_probs_connecting_node()

            node = node.parent

        # assert that connecting node is always the same as root
        while subtree.connecting_node is not None:
            root = SubTreeNode(subtree, 0)
            assert root.get_value() == subtree.connecting_node.get_value(), "Connecting node value should be the same as its parent's value."
            assert root.get_value_eval() == subtree.connecting_node.get_value_eval(), "Connecting node value eval should be the same as its parent's value eval."
            subtree = subtree.connecting_node.subtree


        vals = torch.cat((vals.squeeze(-1), *loop_vals), dim=0)  # shape (n + len(loop_vals),)

        self.min_max_stats.mass_update_tensor(vals)


    def select_child(self, node):
        # argmax with random tie-breaking
        ucb_scores = self.puct.get_ucb_scores(node, self.min_max_stats)
        action = arg_max_with_tie_breaking(ucb_scores)
        #action = 0
        return action, node.get_child(action)


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


    def get_ucb_scores(self, parent, min_max_stats):
        """
        Compute the UCB scores for all children of a parent node.
        """
        ucb_scores = []
        for child in parent.children.values():
            ucb_score = self.ucb_score(parent, child, min_max_stats)
            ucb_scores.append(ucb_score)
        return ucb_scores


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
            child.get_value() if len(self.config.players) == 1 else -child.get_value()
        )
        return raw_value


class PUCT_MVC(PUCT):
    def __init__(self, config):
        super().__init__(config)


    def Q(self, child):
        return child.get_value()

    def U(self, parent, child):
        par_inv_q_var = parent.get_inv_var()
        child_inv_q_var = child.get_inv_var()

        return self.config.PUCT_C * (math.sqrt(par_inv_q_var) / (child_inv_q_var + 1))



class PUCT_Subtree():
    def __init__(self, config, device):
        self.device = device
        self.config = config


    def get_ucb_scores(self, parent, min_max_stats):

        # these should both be torch.Tensors (parent_inv_var is a scalar tensor)
        parent_inv_var = parent.get_inv_var()  # shape: ()
        children_inv_var = parent.get_children_inv_vars(include_self=False)  # shape: (n_children,)

        child_priors = parent.get_prior().unsqueeze(1)  # shape: (n_children,)

        # vectorized U computation
        # sqrt(parent_inv_var) broadcasts over the children dimension
        U_values = child_priors * self.config.PUCT_C * (parent_inv_var.sqrt() / (children_inv_var + 1))

        Q_values = parent.get_children_vals(include_self=False)
        # Normalize Q values using min-max stats
        Q_values = min_max_stats.normalize(Q_values)

        # Combine U and Q values
        ucb_scores = Q_values + U_values
        return ucb_scores.squeeze()  # shape: (n_children,)



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

    def mass_update(self, values):
        self.maximum = max(self.maximum, max(values))
        self.minimum = min(self.minimum, min(values))


    def mass_update_tensor(self, vals):
        self.maximum = max(self.maximum, vals.max().item())
        self.minimum = min(self.minimum, vals.min().item())


    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value



    #
    # def backpropagate_more(self, node_list, value_list, to_play, min_max_stats):
    #     # put parents in a dict
    #
    #     def updates(node):
    #         # node.increment_visit_count()
    #         min_max_stats.update(value)
    #         node.variance = None
    #         node.policy_value = None
    #
    #     parents = defaultdict(list)
    #     # reverse traversal to get children first
    #     for i in range(len(node_list) - 1, -1, -1):
    #         node = node_list[i]
    #         value = value_list[i].item()
    #
    #         child_val_sum, child_rew_sum, visits = parents.get(node, (0, 0, 0))
    #         # add own reward for parent pluts the discounted child rewards
    #         rew_sum = node.reward * (visits + 1) + self.config.discount * child_rew_sum # correct
    #         val_sum = self.config.discount * (value + child_val_sum)
    #
    #         # if parent not in dict, add it (todo check if this is really q)
    #         (sibling_val_cum_sum, sibling_rew_cum_sum, sibling_cum_visits) = parents.get(node.parent, (0, 0, 0))
    #         parents[node.parent] = (sibling_val_cum_sum + val_sum, sibling_rew_cum_sum + rew_sum, sibling_cum_visits + visits + 1)
    #
    #         node.value_sum += (value + child_rew_sum + child_val_sum)
    #         # assert that value_sum_2 equals value sum (to a few decimals)
    #
    #         updates(node)




    # def make_causal_mask_fast(self, seqs, device):
    #     n = len(seqs)
    #     # Convert sequences to tuples for quicker prefix comparisons
    #     seqs = [tuple(seq) for seq in seqs]
    #     lengths = [len(s) for s in seqs]
    #
    #     # We'll build a boolean tensor of shape [n, n] for the allowed positions
    #     # and incorporate the causal condition (j <= i) in a single loop.
    #     allowed_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    #
    #     for i in range(n):
    #         seq_i = seqs[i]
    #         len_i = lengths[i]
    #         # Only check j up to i for the causal condition
    #         for j in range(i + 1):
    #             seq_j = seqs[j]
    #             len_j = lengths[j]
    #             # Check that seq_j is a prefix of seq_i
    #             if len_j <= len_i and seq_i[:len_j] == seq_j:
    #                 allowed_mask[i, j] = True
    #
    #     # Now include the "global" token at index 0 in a new (n+1) x (n+1) matrix
    #     new_allowed = torch.zeros((n + 1, n + 1), dtype=torch.bool, device=device)
    #     # Allow the global token to attend only to itself
    #     new_allowed[0, 0] = True
    #     # All real tokens (1..n) can always attend to the global token (column 0)
    #     new_allowed[1:, 0] = True
    #     # Copy in our computed mask
    #     new_allowed[1:, 1:] = allowed_mask
    #
    #     # Convert boolean allowed positions to the float mask format (0 for allowed, -inf for disallowed)
    #     attention_mask = torch.where(
    #         new_allowed,
    #         torch.tensor(0.0, device=device),
    #         torch.tensor(float('-inf'), device=device)
    #     )
    #     return attention_mask