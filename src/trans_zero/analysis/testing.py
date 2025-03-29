from trans_zero.core.self_play import SelfPlay
import torch

from trans_zero.paths import PROJECT_ROOT
from trans_zero.utils import models
import trans_zero.networks.muzero_network as mz_net

import json


def seq_testing(muzero, file, results_file):
    from trans_zero.core.self_play import update_pred_dict

    model = mz_net.MuZeroNetwork(muzero.config)
    try:
        model.set_weights(muzero.checkpoint["weights"])
    except Exception as e:
        #print(f"Error: {e}")
        print(f"\ntrying new weights\n")
        model.set_weights(SelfPlay.remove_module_prefix(muzero.checkpoint["weights"]))
    model.eval()

    all_runs_dicts = []

    with open(file, 'r') as f:
        preds = [json.loads(line) for line in f.readlines()]
    preds = [pred['results'] for pred in preds]

    def get_obs_as_pair(all_runs):
        def get_single_pair(run):
            observations = [pred['observation'] for pred in run]
            predictions = [pred['predictions'] for pred in run]
            actions_sequences = [[pred['as'] for pred in preds] for preds in predictions]
            return [{"obs": obs, "as": as_seq} for obs, as_seq in zip(observations, actions_sequences)]

        return [get_single_pair(run) for run in all_runs]

    all_runs = get_obs_as_pair(preds)
    for i, run in enumerate(all_runs):
        print(f"Run {i}")
        run_dict = {
            "game": 0,
            "results": []
        }

        for step in run:
            obs_ar = step['obs']
            pred_dict = {
                "observation": obs_ar,
                "predictions": []
            }

            obs = (
                torch.tensor(obs_ar)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            ).unsqueeze(0)

            (
                value,
                reward,
                policy_logits,
                encoded_state,
            ) = model.initial_inference(obs)
            value = models.support_to_scalar(value, muzero.config.support_size).item()
            reward = models.support_to_scalar(reward, muzero.config.support_size).item()

            update_pred_dict(pred_dict, value, reward, policy_logits, [], [0,1,2])

            for actions in step['as']:
                if actions == []:
                    continue
                policy_logits, value, reward = model.prediction(encoded_state, torch.tensor([actions]))
                value = models.support_to_scalar(value, muzero.config.support_size).item()
                reward = models.support_to_scalar(reward, muzero.config.support_size).item()

                update_pred_dict(pred_dict, value, reward, policy_logits, actions,[0,1,2])

            run_dict['results'].append(pred_dict)
        all_runs_dicts.append(run_dict)


    with open(results_file, "w") as f:
        for game_dict in all_runs_dicts:
            json.dump(game_dict, f)
            f.write('\n')  # Add a newline after each JSON object


    return all_runs_dicts



def visualize_model(muzero):
    from torchviz import make_dot
    network = "double"
    muzero.config.network = network

    model = mz_net.MuZeroNetwork(muzero.config)

    batch_size = 1
    observation = torch.randn(batch_size, *muzero.config.observation_shape)  # Example observation
    action = torch.tensor([[0]])
    _,_,_, hidden_state = model.initial_inference(observation)

    if network == "double":
        org_hs = hidden_state
        hidden_state, trans_hidden_state = torch.chunk(hidden_state, 2, dim=0)

    if network == "fully_connected":
        value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action)
    else:
        value, reward, policy_logits, _ = model.recurrent_inference(hidden_state, action, hidden_state, action)

    if network == "double":
        hidden_state = org_hs

    dot_representation = make_dot((value, reward, policy_logits, hidden_state), params=dict(model.named_parameters()))
    dot_representation.render(f"graphs/representation_graph_{network}_shared", format="png")

def setup_testing(muzero, args):
    if args.model_path is not None:
        muzero.load_model(checkpoint_path=args.model_path)

    if args.test_mode == "seq":
        print("seq testing")
        seq_file = f"{PROJECT_ROOT}/data/predictions/preds/5x5_res_ra_1000_old.json"
        results_file = f"{PROJECT_ROOT}/data/predictions/double_preds/5x5_trans_on_res_ra_1000.json"
        seq_testing(muzero, seq_file, results_file)

    elif args.test_mode == "viz":
        print("vizualizing")
        visualize_model(muzero)
    elif args.test_mode == "n_maps":
        print("more map testing")
        muzero.config.show_preds = True
        name = "5x5_res_ra_1000_new.json"
        muzero.config.preds_file = f"{PROJECT_ROOT}/data/predictions/preds/{name}"

        #for i in range(3):
        results = muzero.test(render=False, opponent="self", muzero_player=None, num_tests=1000)
        # put results into file
        with open(f"{PROJECT_ROOT}/data/predictions/results/{name}", "w") as f:
            json.dump(results, f)

    else:
        results = muzero.test(render=True, opponent="self", muzero_player=None)
        print(results)
        print(f"total reward: {sum(results[0])}")