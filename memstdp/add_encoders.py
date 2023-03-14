from bindsnet.memstdp import add_encodings


class Encoder:
    # language=rst
    """
    Base class for spike encodings transforms.

    Calls ``self.enc`` from the subclass and passes whatever arguments were provided.
    ``self.enc`` must be callable with ``torch.Tensor``, ``*args``, ``**kwargs``
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class RankOrderTTFSEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable RankOrderEncoder which encodes as defined in
        :code:`bindsnet.encoding.rank_order`

        :param time: Length of RankOrder spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = add_encodings.rank_order_TTFS


class RankOrderTTASEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable RankOrderEncoder which encodes as defined in
        :code:`bindsnet.encoding.rank_order`

        :param time: Length of RankOrder spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = add_encodings.rank_order_TTAS


class LinearRateEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable RankOrderEncoder which encodes as defined in
        :code:`bindsnet.encoding.rank_order`

        :param time: Length of RankOrder spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = add_encodings.linear_rate
