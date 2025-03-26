import math
import os
import time
import json

import numpy
import ray
import torch

import models
import networks.muzero_network as mz_net
from value_utils.policies import MinimalVarianceConstraintPolicy
from value_utils.utility_functions import policy_value, compute_inverse_q_variance, get_children_inverse_variances


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        if (hasattr(self.config, "game_name") and self.config.game_name == "frozen_lake"
                or self.config.game_name == "custom_grid"
                or "lunarlander" in self.config.game_name):
            self.game = Game(seed, config=config)
        else:
            self.game = Game()

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = mz_net.MuZeroNetwork(self.config)
        try:
            self.model.set_weights(initial_checkpoint["weights"])
        except Exception:
            print(f"trying new weights")
            self.model.set_weights(self.remove_module_prefix(initial_checkpoint["weights"]))

        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        print(f"Using {'cuda' if self.config.selfplay_on_gpu else 'cpu'} for self-play.")
        self.model.eval()

        if self.config.action_selection == "mvc":
            self.mvc = MinimalVarianceConstraintPolicy(config=self.config)

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        game_number = 0
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            game_number += 1
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                    game_number
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                    game_number
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player, game_number
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """

        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        if self.config.show_preds:
            game_dict = {"game": game_number, "results": [], "chosen_actions": []}

        with torch.no_grad():

            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensional. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play(): # note to Emil, where you go when using frozen_lake
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        temperature != 0
                    )

                    if self.config.action_selection == "mvc":
                        root.reset_var_val()

                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")

                    if self.config.show_preds:
                        game_dict["results"].append(mcts_info["predictions"])
                        game_dict["chosen_actions"].append(int(action))

                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)


                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    print(f"Obtained reward: {reward}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        if self.config.show_preds:
            # append to file that is made if it does not exist
            file_path = self.config.preds_file
            with open(file_path, "a") as f:
                json.dump(game_dict, f)
                f.write('\n')  # Add a newline after each JSON object
                f.flush()

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )


    @staticmethod
    def std_action_selection(node, temperature):
        """
                Select action according to the visit count distribution and the temperature.
                The temperature is changed dynamically with the visit_softmax_temperature function
                in the config.
                """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


    def mvc_action_selection(self, tree):
        policy_dist = self.mvc.softmaxed_distribution(tree)
        action = policy_dist.sample().item()
        return action


    def select_action(self, node, temperature):
        if self.config.action_selection == "mvc":
            return self.mvc_action_selection(node)
        else:
            return self.std_action_selection(node, temperature)


    @staticmethod
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # Check if ".module." appears in the key
            if ".module." in k:
                new_key = k.replace(".module.", ".")
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict


def update_pred_dict(pred_dict, value, reward, policy_logits, action_sequence, action_space):
    as_dict = {
        0: 'L',
        1: 'R',
        2: 'F'
    }

    policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in action_space]), dim=0
    ).tolist()


    pred_dict["predictions"].append(
        {

            "v": value,
            "r": reward,
            "p": policy_values,

            "as": action_sequence,

        }
    )


def update_pred_dict_double(pred_dict, value, reward, policy_logits, trans_value, trans_reward, trans_policy_logits, action_sequence, support_size):


    pred_dict["predictions"].append(
        {
            "v": value,
            "r": reward,
            "p": policy_logits,

            "tv": models.support_to_scalar(trans_value, support_size).item(),
            "tr": models.support_to_scalar(trans_reward, support_size).item(),
            "tp": trans_policy_logits.tolist(),

            "as": action_sequence,
        }
    )

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
        if 'mvc' in [self.config.PUCT_Q, self.config.PUCT_U, self.config.action_selection]:
            self.policy = MinimalVarianceConstraintPolicy(config=self.config)

        import time
        timestamp = time.time()
        self.file_path = f"test_scores/test_ucb_scores_{timestamp}.csv"

        if self.config.test_ucb:
            self.min_max_stats_std = MinMaxStats()
            with open(self.file_path, "w") as f:
                f.write("P,C,UCB,UCB_mvc,U,U_mvc,Q,Q_mvc\n")
                f.flush

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
        is_trans_net = "trans" in self.config.network # todo better implementation later
        is_double_net = self.config.network == "double"

        if self.config.show_preds:
            pred_dict = {
                "observation": observation.squeeze().tolist(),
                "predictions": []
            }

        # print(f"is trans net: {is_trans_net}")
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            if 'mvc' in [self.config.action_selection, self.config.PUCT_Q, self.config.PUCT_U]:
                root = MVCNode(0, use_reward=self.config.predict_reward, parent=None, action_space_size=len(self.config.action_space))
            else:
                root = Node(0, use_reward=self.config.predict_reward)
            # Observation in right shape
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

            if is_double_net:
                # value is first half
                root_predicted_value, root_predicted_trans_value = root_predicted_value.chunk(2, dim=0)
                policy_logits, trans_policy_logits = policy_logits.chunk(2, dim=0)
                reward, trans_reward = reward.chunk(2, dim=0)
                hidden_state, trans_root_hidden_state = hidden_state.chunk(2, dim=0)

            # Make the root predicted value and reward a scalar
            root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()

            if self.config.show_preds:
                if is_double_net:
                    update_pred_dict_double(pred_dict, root_predicted_value, reward, policy_logits, root_predicted_trans_value, trans_reward, trans_policy_logits, [], self.config.support_size)
                else:
                    update_pred_dict(pred_dict, root_predicted_value, reward, policy_logits, [], self.config.action_space)

            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            root.expand(
                legal_actions,
                to_play,
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            )


        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            actions = []
            current_tree_depth = 0

            if self.config.PUCT_U == "mvc":
                node.reset_var()

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats) # tree action selection
                search_path.append(node)
                actions.append(action)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            if is_trans_net or is_double_net:
                root_hidden_state = root.hidden_state if not is_double_net else trans_root_hidden_state

                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[actions[-1]]]).to(root_hidden_state.device),
                    root_hidden_state = root_hidden_state,
                    action_sequence= torch.tensor([actions]).to(root_hidden_state.device),
                )

            else:
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[action]]).to(parent.hidden_state.device),
                )

            if is_double_net:
                value, trans_value = value.chunk(2, dim=0)
                policy_logits, trans_policy_logits = policy_logits.chunk(2, dim=0)
                reward, trans_reward = reward.chunk(2, dim=0)

            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()


            if self.config.show_preds:
                if is_double_net:
                    update_pred_dict_double(pred_dict, value, reward, policy_logits, trans_value, trans_reward, trans_policy_logits,
                                 [int(a) for a in actions], self.config.support_size)
                else:
                    update_pred_dict(pred_dict, value, reward, policy_logits, [int(a) for a in actions], self.config.action_space)

            node.expand(
                self.config.action_space,
                virtual_to_play,
                value,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        if self.config.show_preds:
            extra_info["predictions"] = pred_dict
        return root, extra_info

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

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score without recalculating.
        """
        # Calculate scores once, store action-score pairs
        # if self.config.PUCT_U == "mvc":
        #     node.reset_var()

        action_score_pairs = [
            (action, self.ucb_score(node, child, min_max_stats))
            for action, child in node.children.items()
        ]

        if self.config.test_ucb:
            self.write_test_to_file(node, min_max_stats)

        # Find the maximum UCB score
        max_ucb = max(score for action, score in action_score_pairs)

        # Select actions that have the maximum UCB score
        best_actions = [action for action, score in action_score_pairs if score == max_ucb]

        # Randomly select among best actions
        selected_action = numpy.random.choice(best_actions)

        return selected_action, node.children[selected_action]



    def calc_U(self, parent, child):
        if self.config.PUCT_U == "std":
            U = self.calc_U_std(parent, child)
        elif self.config.PUCT_U == "mvc":
            U = self.calc_U_mvc(parent, child)
        elif self.config.PUCT_U == "mvc_exp":
            U = self.calc_U_mvc_experimental(parent, child)
        else:
            raise NotImplementedError("Action selection policy not implemented.")

        return child.prior * U

    def calc_U_std(self, parent, child):
        pb_c_log = math.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c = pb_c_log + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return pb_c


    def calc_U_mvc(self, parent, child):
        par_inv_q_var = compute_inverse_q_variance(parent, self.policy, self.config.discount)
        child_inv_q_var = compute_inverse_q_variance(child, self.policy, self.config.discount)

        return self.config.PUCT_C * math.sqrt(par_inv_q_var) / (child_inv_q_var + 1)


    # todo try
    def calc_U_mvc_experimental(self, parent, child):
        par_inv_q_var = compute_inverse_q_variance(parent, self.policy, self.config.discount)
        child_inv_q_var = compute_inverse_q_variance(child, self.policy, self.config.discount)

        pb_c_log = math.log(
            (par_inv_q_var + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c = pb_c_log + self.config.pb_c_init
        pb_c *= math.sqrt(par_inv_q_var) / (child_inv_q_var + 1)
        return pb_c



    def calc_Q(self, child, min_max_stats):

        if self.config.PUCT_Q == "std":
            return self.calc_Q_std(child, min_max_stats)
        elif self.config.PUCT_Q == "mvc":
            return self.calc_Q_mvc(child, min_max_stats)
        else:
            raise NotImplementedError("Action selection policy not implemented.")


    def calc_Q_mvc(self, child, min_max_stats):
        return min_max_stats.normalize(policy_value(child, self.policy, self.config.discount))


    def calc_Q_std(self, child, min_max_stats):
        if child.visit_count < 1:
            return 0

        raw_value = child.reward + self.config.discount * (
            child.value() if len(self.config.players) == 1 else -child.value()
        )
        return min_max_stats.normalize(raw_value) # TODO add to mvc implementation also


    def ucb_score(self, parent, child, min_max_stats):
        """
        Compute the PUCT (Predictor + UCT) score for a child node.
        Score = Q + U
        Where:
            - Q is the normalized value estimate
            - U is the exploration bonus
        """

        # --- 1. Calculate C (dynamic exploration constant) ---
        U = self.calc_U(parent, child)

        # --- 3. Calculate Q (normalized value score) ---
        Q = self.calc_Q(child, min_max_stats)

        # --- 5. Combine to get final PUCT score: UCB = Q + U ---
        return Q + U

    def ucb_score_test(self, parent, child, min_max_stats):
        u_std, u_mvc = self.calc_U_std(parent, child), self.calc_U_mvc(parent, child)
        q_std, q_mvc = self.calc_Q_std(child, self.min_max_stats_std), self.calc_Q_mvc(child, min_max_stats)
        return u_std, u_mvc, q_std, q_mvc


    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                if self.config.PUCT_Q == "std":
                    value = node.reward + self.config.discount * value
                # todo Emil, seems, correct, verify
                elif self.config.PUCT_Q == "mvc":
                    value = policy_value(node, self.policy, self.config.discount)
                min_max_stats.update(value)

                # todo emil remove at some point when tested
                if self.config.test_ucb:
                    self.min_max_stats_std.update(node.reward + self.config.discount * value)

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")

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


class Node:

    def __init__(self, prior, name="root", use_reward=True):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.value_evaluation = 0
        self.use_reward = use_reward

        # for debugging
        self.name = name

    def expanded(self):
        return len(self.children) > 0

    def make_child(self, prior, child_name):
        """Factory method to create a child node."""
        return Node(prior, name=child_name, use_reward=self.use_reward)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, value, reward, policy_logits, hidden_state):
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

    def __repr__(self):
        return self.name


class MVCNode(Node):
    def __init__(self, prior, action_space_size, parent=None, name="root", use_reward=True):
        super().__init__(prior, name, use_reward)
        self.parent = parent
        self.variance = None
        self.policy_value = None
        self.action_space_size = action_space_size

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


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations

    def get_forward_stacked_observations(self, index, num_stacked_observations):
        """
        Stack the observation at `index` and the next `num_stacked_observations`
        along a new axis. If there are not enough future observations, pad with zeros.

        This version preserves the original channel dimension. For example, if
        observations are (1, H, W), the output will be (num_stacked+1, 1, H, W).
        """
        index = index % len(self.observation_history)
        obs_list = []
        # Include the starting observation and future observations.
        for offset in range(num_stacked_observations + 1):
            pos = index + offset
            if pos < len(self.observation_history):
                obs_list.append(torch.from_numpy(self.observation_history[pos].copy()))
            else:
                obs_list.append(torch.zeros_like(torch.from_numpy(self.observation_history[index].copy())))
                # torch version of exact same
                # obs_list.append(torch.zeros_like(self.observation_history[index

        # Use np.stack to create a new axis for the time dimension.
        stacked_observations = torch.stack(obs_list, dim=0)

        return stacked_observations


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
