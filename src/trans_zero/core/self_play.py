import math
import time
import json

import numpy
import ray
import torch

from trans_zero.utils import models
import trans_zero.networks.muzero_network as mz_net
from trans_zero.mvc_utils.policies import MinimalVarianceConstraintPolicy
from .mcts import MCTS, MCTS_PLL, MCTS_MVC

@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed, config=config)

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

        if self.config.action_selection == "mvc": # todo
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


            # normal train mode
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
                if opponent == "self" or muzero_player == self.game.to_play():

                    if self.config.expansion_strategy == 'deep':
                        root, mcts_info = MCTS_PLL(self.config).run(
                            self.model,
                            stacked_observations,
                            self.game.legal_actions(),
                            self.game.to_play(),
                            temperature != 0
                        )

                    else:# note to Emil, where you go when using frozen_lake
                        if self.config.PUCT_variant == "mvc":
                            root, mcts_info = MCTS_MVC(self.config).run(
                                self.model,
                                stacked_observations,
                                self.game.legal_actions(),
                                self.game.to_play(),
                                temperature != 0
                            )
                        else:
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
            [child.get_visit_count() for child in node.children.values()], dtype="int32"
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
            sum_visits = sum(child.get_visit_count() for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].get_visit_count() / sum_visits
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


