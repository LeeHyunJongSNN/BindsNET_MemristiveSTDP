from typing import Iterable, Iterator, Optional, Union

import torch

from bindsnet.memstdp.add_encodings import rank_order_TTFS, linear_rate

def rank_order_TTFS_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        # Encode datum as rank order-encoded spike trains.
        yield rank_order_TTFS(datum=data[i], time=time, dt=dt)


def rank_order_TTAS_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        # Encode datum as rank order-encoded spike trains.
        yield rank_order_TTAS(datum=data[i], time=time, dt=dt)


def linear_rate_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        # Encode datum as rank order-encoded spike trains.
        yield linear_rate(datum=data[i], time=time, dt=dt)
