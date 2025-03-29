import nevergrad
import ray
import torch
from trans_zero.core.muzero import MuZero


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests, logger="tensorboard"
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train()
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train()
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value