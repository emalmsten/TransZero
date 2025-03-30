import torch
import math
import ray
import trans_zero.networks.muzero_network as mz_net

@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = mz_net.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def calc_num_gpus(config, split_resources_in):
    total_gpus = calc_total_gpus(config)
    num_gpus = total_gpus / split_resources_in
    if 1 < num_gpus:
        num_gpus = math.floor(num_gpus)

    return num_gpus, total_gpus


def calc_total_gpus(config):
    # Manage GPUs
    if config.max_num_gpus == 0 and (
            config.selfplay_on_gpu
            or config.train_on_gpu
            or config.reanalyse_on_gpu
    ):
        raise ValueError(
            "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
        )

    if (
            config.selfplay_on_gpu
            or config.train_on_gpu
            or config.reanalyse_on_gpu
    ):
        total_gpus = (
            config.max_num_gpus
            if config.max_num_gpus is not None
            else torch.cuda.device_count()
        )
    else:
        total_gpus = 0

    return total_gpus

def calc_num_gpus_per_worker(num_gpus, config, logger=None):
    if 0 < num_gpus:
        num_gpus_per_worker = num_gpus / (
                config.train_on_gpu
                + config.num_workers * config.selfplay_on_gpu
                + (logger is not None) * config.selfplay_on_gpu
                + config.use_last_model_value * config.reanalyse_on_gpu
        )
        if 1 < num_gpus_per_worker:
            num_gpus_per_worker = math.floor(num_gpus_per_worker)
    else:
        num_gpus_per_worker = 0

    return num_gpus_per_worker