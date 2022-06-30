from typing import Optional

import torch

def rank_order_TTFS(
    datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a rank order coding-like representation.
    Temporally ordered by decreasing intensity. Auxiliary spikes can appear. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Create spike times in order of decreasing intensity.
    datum /= datum.max()
    times = torch.zeros(size)
    times[datum != 0] = 1 / datum[datum != 0]
    times *= time / times.max()  # Extended through simulation time.
    times = torch.ceil(times).long()

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    term = 5
    for i in range(size):
        if 0 < times[i] < time:
            spikes[times[i] - 1, i] = 1
            for j in range(times[i], time):
                if j % term == 0:
                    aux = j + (times[i] - 1) % term
                    if aux < time:
                       spikes[aux, i] = 1


    return spikes.reshape(time, *shape)