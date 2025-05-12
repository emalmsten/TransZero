def set_global_seeds(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import torch

def time_cuda_op(name, op,  *args, warmup=10, runs=100, **kwargs):
    """
    Time a CUDA operation using torch.cuda.Event.

    Args:
        op (callable): The operation/function to run.
        *args: Positional arguments to pass to the op.
        warmup (int): Number of warm-up runs (not timed).
        runs (int): Number of timed runs.
        **kwargs: Keyword arguments to pass to the op.

    Returns:
        float: Average execution time in milliseconds.
    """
    # Warm-up (not timed)
    for _ in range(warmup):
        _ = op(*args, **kwargs)

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0.0
    for _ in range(runs):
        start_event.record()
        _ = op(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)

    avg_time_ms = total_time / runs
    print(f"Average time for {name}: {avg_time_ms:.5f} ms over {runs} runs")
    return avg_time_ms
