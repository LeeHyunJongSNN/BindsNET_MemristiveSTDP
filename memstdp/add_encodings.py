from typing import Optional

import torch

def rank_order_TTFS(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
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
    term = 2
    for i in range(size):
        if 0 < times[i] < time:
            spikes[times[i] - 1, i] = 1
            for j in range(times[i], time):
                if j % term == 0:
                    aux = j + (times[i] - 1) % term
                    if aux < time:
                       spikes[aux, i] = 1

    return spikes.reshape(time, *shape)


def rank_order_TTAS(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
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
    term = 10       # 10
    jitter = 2      # 2
    num = 10        # 10
    for i in range(size):
        if 0 < times[i] < time:
            for k in range(1, num + 1):
                spikes[times[i] - jitter * k, i] = 1
            for j in range(times[i], time):
                if j % term == 0:
                    aux = j + (times[i] - 1) % term
                    if aux < time:
                        spikes[aux, i] = 1

    return spikes.reshape(time, *shape)


def linear_rate(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    :param datum: Repeats a tensor along a new dimension in the 0th position for
        ``int(time / dt)`` timesteps.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of repeated data along the 0-th
        dimension.
    """
    time = int(time / dt)
    b = -2
    shape, size = datum.shape, datum.numel()

    converted = (30 * datum + b ).view(-1)
    for i in range(len(converted)):
        if converted[i].item() < 13:
            converted[i] = 0

    spikes = torch.zeros(time, size).byte()
    for k in range(time):
        for j in range(size):
            if converted[j] != 0:
                if k % (round(time / converted[j].item())) == 0:
                    spikes[k, j] = 1

    return spikes.reshape(time, *shape)

