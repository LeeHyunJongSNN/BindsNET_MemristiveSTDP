import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from scipy.stats import bernoulli
from sklearn.preprocessing import minmax_scale

from bindsnet.utils import im2col_indices
from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractConnection,
    Connection,
    Conv1dConnection,
    Conv2dConnection,
    Conv3dConnection,
    LocalConnection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
)

class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, (float, int)):
            nu = [nu, nu]

        self.nu = torch.zeros(2, dtype=torch.float)
        self.nu[0] = nu[0]
        self.nu[1] = nu[1]

        if (self.nu == torch.zeros(2)).all() and not isinstance(self, NoOp):
            warnings.warn(
                f"nu is set to [0., 0.] for {type(self).__name__} learning rule. "
                + "It will disable the learning process."
            )

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            if self.source.batch_size == 1:
                self.reduction = torch.squeeze
            else:
                self.reduction = torch.sum
        else:
            self.reduction = reduction

        # Weight decay.
        self.weight_decay = 1.0 - weight_decay if weight_decay else 1.0

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        # Bound weights.
        if (
            self.connection.wmin != np.inf or self.connection.wmax != -np.inf
        ) and not isinstance(self, NoOp):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)

class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        super().update()


class MemristiveSTDP_Simplified(LearningRule):
    # language=rst
    """
    This rule is simplified STDP with memristive characteristic.
    It involves both pre-synaptic and post-synaptic spiking activity.
    By default, pre-synaptic update is LTD and the post-synaptic update is LTP.
    This rule doesn't allow input neurons' spiking proportion to affect synaptic weights regulation.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MemristiveSTDP_Simplified`` learning rule.
        :param connection: An ``AbstractConnection`` object whose weights the
            ``MemristiveSTDP_Simplifeid`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self ,**kwargs) -> None:
        # language=rst
        """
        Simplified Memristive STDP learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Information of input and excitatory spikes
        source_s = self.source.s.view(-1).long()
        target_s = self.target.s.view(-1).long()
        X_size = torch.numel(source_s)
        Ae_size = torch.numel(target_s)

        # Spike recording variables
        s_record = kwargs.get('s_record', [])
        t_record = kwargs.get('t_record', [])
        simulation_time = kwargs.get('simulation_time')
        s_record.append(source_s.tolist())
        t_record.append(target_s.tolist())
        source_r = torch.tensor(s_record)
        target_r = torch.tensor(t_record)

        # Time variables
        time = len(source_r)
        pulse_time_LTP = 45  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 45  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        X_cause_index = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Synaptic configurations
        drop_index_input = []
        template_exc = []
        reinforce_index_input = []

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        template_exc = kwargs.get('template_exc')  # ST excitatory neuron num
        ST = kwargs.get('ST')  # ST useage
        ADC = kwargs.get('ADC')  # ADC useage
        DS = kwargs.get("DS")  # DS simulation

        # Random Conductance uperbound and underbound
        if grand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_index_input = kwargs.get('drop_index_input')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            for i in range(len(template_exc)):
                for j in drop_index_input[i]:
                    self.connection.w[j, template_exc[i]] = 0
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, template_exc[i]] <= gmax[j, template_exc[i]] * 0.4:
                        self.connection.w[j, template_exc[i]] = gmax[j, template_exc[i]] * \
                                                                reinforce_ref[int(template_exc[i])][
                                                                    int(np.where(
                                                                        j == reinforce_index_input[i])[
                                                                            0])] * 0.5


        # Dead synpase simulation
        if DS:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')
            for i in range(len(dead_index_exc)):
                for j in dead_index_input[i]:
                    self.connection.w[j, dead_index_exc[i]] = 0


        # Weight update with memristive characteristc
        if vltp == 0 and vltd == 0:  # Fully linear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(-1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += b * (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += b * (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        for k in Ae_index_LTD:
                                            self.connection.w[i, k.item()] -= (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[l])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[l]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            for k in Ae_index_LTD:
                                                self.connection.w[i, k.item()] -= (gmax[i, k.item()] - gmin[
                                                    i, k.item()]) / 256

        elif vltp != 0 and vltd == 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(-1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                        i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                        i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        for k in Ae_index_LTD:
                                            self.connection.w[i, k.item()] -= (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[l])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[l]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            for k in Ae_index_LTD:
                                                self.connection.w[i, k.item()] -= (gmax[i, k.item()] - gmin[
                                                    i, k.item()]) / 256

        elif vltp == 0 and vltd != 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(-1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += b * (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += b * (gmax[i, k.item()] - gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        for k in Ae_index_LTD:
                                            self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] + g1ltd[
                                                i, k.item()] - gmax[i, k.item()]) * (1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[l])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[l]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            for k in Ae_index_LTD:
                                                self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                   g1ltd[
                                                                                       i, k.item()] - gmax[
                                                                                       i, k.item()]) * (
                                                                                              1 - np.exp(vltd / 256))

        elif vltp != 0 and vltd != 0:  # Fully nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(-1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                        i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                for k in Ae_index_LTP:
                                    self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                        i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        for k in Ae_index_LTD:
                                            self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] + g1ltd[
                                                i, k.item()] - gmax[i, k.item()]) * (1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[l])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[l]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            for k in Ae_index_LTD:
                                                self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                   g1ltd[
                                                                                       i, k.item()] - gmax[
                                                                                       i, k.item()]) * (
                                                                                              1 - np.exp(vltd / 256))


        # Adapitve Drop Connect
        if ADC:
            p = np.round(minmax_scale(
                np.nan_to_num(self.connection.w.cpu().detach().numpy().reshape(X_size * Ae_size), copy=False),
                feature_range=(0.9999, 1)).reshape(X_size, Ae_size), 3)
            m = torch.zeros(X_size, Ae_size).to('cuda')

            for i in range(X_size):
                for j in range(Ae_size):
                    m[i, j] = int(bernoulli.rvs(p[i, j], size=1))

            self.connection.w *= m


        super().update()


class MemristiveSTDP(LearningRule):
    # language=rst
    """
    This rule is STDP with memristive characteristic.
    It involves both pre-synaptic and post-synaptic spiking activity.
    By default, pre-synaptic update is LTD and the post-synaptic update is LTP.
    Input neurons' spiking proportion affects synaptic weight regulation.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MemristiveSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MemristiveSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self ,**kwargs) -> None:
        # language=rst
        """
        Memristive STDP learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Information of input and excitatory spikes
        source_s = self.source.s.view(-1).long()
        target_s = self.target.s.view(-1).long()
        X_size = torch.numel(source_s)
        Ae_size = torch.numel(target_s)

        # Spike recording variables
        s_record = kwargs.get('s_record', [])
        t_record = kwargs.get('t_record', [])
        simulation_time = kwargs.get('simulation_time')
        s_record.append(source_s.tolist())
        t_record.append(target_s.tolist())
        source_r = torch.tensor(s_record)
        target_r = torch.tensor(t_record)

        # Time variables
        time = len(source_r)
        pulse_time_LTP = 50  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 50  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        X_cause_index = 0
        X_cause_count = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Synaptic configurations
        drop_index_input = []
        template_exc = []
        reinforce_index_input = []

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        template_exc = kwargs.get('template_exc')  # ST excitatory neuron num
        ST = kwargs.get('ST')  # ST useage
        ADC = kwargs.get('ADC')  # ADC useage
        DS = kwargs.get("DS")  # DS simulation

        # Random Conductance uperbound and underbound
        if grand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_index_input = kwargs.get('drop_index_input')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            for i in range(len(template_exc)):
                for j in drop_index_input[i]:
                    self.connection.w[j, template_exc[i]] = 0
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, template_exc[i]] <= gmax[j, template_exc[i]] * 0.4:
                        self.connection.w[j, template_exc[i]] = gmax[j, template_exc[i]] * \
                                                                reinforce_ref[int(template_exc[i])][
                                                                    int(np.where(
                                                                        j == reinforce_index_input[i])[
                                                                            0])] * 0.5


        # Dead synpase simulation
        if DS:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')
            for i in range(len(dead_index_exc)):
                for j in dead_index_input[i]:
                    self.connection.w[j, dead_index_exc[i]] = 0


        # Weight update with memristive characteristc
        if vltp == 0 and vltd == 0:  # Fully linear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                                   gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                                       gmin[i, k.item()]) / 256

        elif vltp != 0 and vltd == 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                                   gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                                       gmin[i, k.item()]) / 256

        elif vltp == 0 and vltd != 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                    g1ltd[i, k.item()] - gmax[
                                                                                        i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                        g1ltd[i, k.item()] - gmax[
                                                                                            i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))

        elif vltp != 0 and vltd != 0:  # Fully nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        self.connection.w[i, k.item()] += (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                    g1ltd[i, k.item()] - gmax[
                                                                                        i, k.item()]) * (
                                                                                           1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    self.connection.w[i, k.item()] -= (self.connection.w[i, k.item()] +
                                                                                        g1ltd[i, k.item()] - gmax[
                                                                                            i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))


        # Adapitve Drop Connect
        if ADC:
            p = np.round(minmax_scale(
                np.nan_to_num(self.connection.w.cpu().detach().numpy().reshape(X_size * Ae_size), copy=False),
                feature_range=(0.9999, 1)).reshape(X_size, Ae_size), 3)
            m = torch.zeros(X_size, Ae_size).to('cuda')

            for i in range(X_size):
                for j in range(Ae_size):
                    m[i, j] = int(bernoulli.rvs(p[i, j], size=1))

            self.connection.w *= m


        super().update()


    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get('random_G')  # Random distribution Gmax and Gmin

        # Random Conductance uperbound and underbound
        if grand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')

        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))

        norm = kwargs.get('norm')

        if vltp == 0 and vltd ==0:
            # LTD
            if self.nu[0].any():
                pre = self.reduction(
                    torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
                )
                update = -(pre.view(self.connection.w.size())
                        * (self.connection.w - (gmax - gmin) / 256) / norm
                )

            # LTP
            if self.nu[1].any():
                post = self.reduction(
                    torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
                )
                update = (post.view(self.connection.w.size())
                        * (self.connection.w - b * (gmax - gmin) / 256) / norm
                )

            self.connection.w += update


        elif vltp != 0 and vltd == 0:
            # LTD
            if self.nu[0].any():
                pre = self.reduction(
                    torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
                )
                update = -(pre.view(self.connection.w.size())
                        * (self.connection.w - (gmax - gmin) / 256) / norm
                )

            # LTP
            if self.nu[1].any():
                post = self.reduction(
                    torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
                )
                update = (post.view(self.connection.w.size())
                        * (g1ltp + gmin - self.connection.w)
                        * (1 - np.exp(-vltp * b / 256)) / norm
                )

            self.connection.w += update


        elif vltp == 0 and vltd != 0:
            # LTD
            if self.nu[0].any():
                pre = self.reduction(
                    torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
                )
                update = -(pre.view(self.connection.w.size())
                        * (g1ltd - gmax + self.connection.w)
                        * (1 - np.exp(vltd / 256)) / norm
                )

            # LTP
            if self.nu[1].any():
                post = self.reduction(
                    torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
                )
                update = (post.view(self.connection.w.size())
                        * (self.connection.w - b * (gmax - gmin) / 256) / norm
                )

            self.connection.w += update


        elif vltp != 0 and vltd != 0:
            # LTD
            if self.nu[0].any():
                pre = self.reduction(
                    torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
                )
                update = -(pre.view(self.connection.w.size())
                        * (g1ltd - gmax + self.connection.w)
                        * (1 - np.exp(vltd / 256)) / norm
                )

            # LTP
            if self.nu[1].any():
                post = self.reduction(
                    torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
                )
                update = (post.view(self.connection.w.size())
                        * (g1ltp + gmin - self.connection.w)
                        * (1 - np.exp(-vltp * b / 256)) / norm
                )

            self.connection.w += update


        super().update()


class MemristiveSTDP_TimeProportion(LearningRule):
    # language=rst
    """
    This rule is STDP with memristive characteristic.
    It involves both pre-synaptic and post-synaptic spiking activity.
    By default, pre-synaptic update is LTD and the post-synaptic update is LTP.
    In addtion, it updates the weight according to the time range between pre-synaptic and post-synaptic spikes.
    Input neurons' spiking proportion affects synaptic weight regulation.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MemristiveSTDP_TimeProportion`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MemristiveSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self ,**kwargs) -> None:
        # language=rst
        """
        Memristive STDP learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Information of input and excitatory spikes
        source_s = self.source.s.view(-1).long()
        target_s = self.target.s.view(-1).long()
        X_size = torch.numel(source_s)
        Ae_size = torch.numel(target_s)

        # Spike recording variables
        s_record = kwargs.get('s_record', [])
        t_record = kwargs.get('t_record', [])
        simulation_time = kwargs.get('simulation_time')
        s_record.append(source_s.tolist())
        t_record.append(target_s.tolist())
        source_r = torch.tensor(s_record)
        target_r = torch.tensor(t_record)

        # Time variables
        time = len(source_r)
        pulse_time_LTP = 50  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 50  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        X_cause_index = 0
        X_cause_count = 0
        X_cause_time = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Synaptic configurations
        drop_index_input = []
        template_exc = []
        reinforce_index_input = []

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        template_exc = kwargs.get('template_exc')  # ST excitatory neuron num
        ST = kwargs.get('ST')  # ST useage
        ADC = kwargs.get('ADC')  # ADC useage
        DS = kwargs.get("DS")  # DS simulation

        # Random Conductance uperbound and underbound
        if grand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_index_input = kwargs.get('drop_index_input')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            for i in range(len(template_exc)):
                for j in drop_index_input[i]:
                    self.connection.w[j, template_exc[i]] = 0
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, template_exc[i]] <= gmax[j, template_exc[i]] * 0.4:
                        self.connection.w[j, template_exc[i]] = gmax[j, template_exc[i]] * \
                                                                    reinforce_ref[int(template_exc[i])][
                                                                        int(np.where(
                                                                            j == reinforce_index_input[i])[
                                                                                0])] * 0.5


        # Dead synpase simulation
        if DS:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')
            for i in range(len(dead_index_exc)):
                for j in dead_index_input[i]:
                    self.connection.w[j, dead_index_exc[i]] = 0     # gmax[j, dead_index_exc[i]]


        # Weight update with memristive characteristc
        if vltp == 0 and vltd == 0:  # Fully linear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[0:Ae_time_LTP, i]).reshape(-1) # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * b * (gmax[i, k.item()] -
                                                                                   gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP, i]).reshape(
                                    -1) # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_time = torch.nonzero(
                                            source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD, i]).reshape(
                                            -1)  # LTP causing spikes time
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                self.connection.w[i, k.item()] -= t * (gmax[i, k.item()] -
                                                                                   gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_time = torch.nonzero(
                                                source_r[Ae_time_LTD:time, i]).reshape(-1)  # LTP causing spikes time
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                    self.connection.w[i, k.item()] -= t * (gmax[i, k.item()] -
                                                                                       gmin[i, k.item()]) / 256

        elif vltp != 0 and vltd == 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[0:Ae_time_LTP, i]).reshape(-1)  # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP, i]).reshape(
                                    -1)  # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_time = torch.nonzero(
                                            source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD, i]).reshape(
                                            -1)  # LTP causing spikes time
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                self.connection.w[i, k.item()] -= t * (gmax[i, k.item()] -
                                                                                   gmin[i, k.item()]) / 256

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_time = torch.nonzero(
                                                source_r[Ae_time_LTD:time, i]).reshape(-1)  # LTP causing spikes time
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                    self.connection.w[i, k.item()] -= t * (gmax[i, k.item()] -
                                                                                       gmin[i, k.item()]) / 256

        elif vltp == 0 and vltd != 0:  # Half nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[0:Ae_time_LTP, i]).reshape(-1)  # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            X_cause_time = torch.nonzero(
                                source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP, i]).reshape(
                                -1)  # LTP causing spikes time
                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                            for k in Ae_index_LTP:
                                for j in range(X_cause_count):
                                    t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                    self.connection.w[i, k.item()] += t * b * (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_time = torch.nonzero(
                                            source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD, i]).reshape(
                                            -1)  # LTP causing spikes time
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                self.connection.w[i, k.item()] -= t * (self.connection.w[i, k.item()] +
                                                                                    g1ltd[i, k.item()] - gmax[
                                                                                        i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_time = torch.nonzero(
                                                source_r[Ae_time_LTD:time, i]).reshape(-1)  # LTP causing spikes time
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                    self.connection.w[i, k.item()] -= t * (self.connection.w[i, k.item()] +
                                                                                        g1ltd[i, k.item()] - gmax[
                                                                                            i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))

        elif vltp != 0 and vltd != 0:  # Fully nonlinear update
            if torch.numel(update_index_and_time) == 0:
                self.connection.w = self.connection.w

            elif torch.numel(update_index_and_time) != 0:
                if torch.numel(torch.nonzero(target_s)) != 0:
                    Ae_time_LTP = time  # Latest update time
                    Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
                if Ae_time_LTP < pulse_time_LTP:
                    if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[0:Ae_time_LTP, i]).reshape(-1)  # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                elif Ae_time_LTP >= pulse_time_LTP:
                    if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                        X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                        [1]].view(
                            -1)  # LTP causing spikes
                        for i in range(X_size):
                            if i in X_cause_index:
                                X_cause_time = torch.nonzero(
                                    source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP, i]).reshape(
                                    -1)  # LTP causing spikes time
                                X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                for k in Ae_index_LTP:
                                    for j in range(X_cause_count):
                                        t = pulse_time_LTP / abs(Ae_time_LTP - X_cause_time[j] + 1)
                                        self.connection.w[i, k.item()] += t * (-self.connection.w[i, k.item()] + g1ltp[
                                            i, k.item()] + gmin[i, k.item()]) * (1 - np.exp(-vltp * b / 256))

                    if time - pulse_time_LTD > 0:
                        if torch.numel(
                                torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                            Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(
                                target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                                X_cause_index = torch.nonzero(
                                    source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_time = torch.nonzero(
                                            source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD, i]).reshape(
                                            -1)  # LTP causing spikes time
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                self.connection.w[i, k.item()] -= t * (self.connection.w[i, k.item()] +
                                                                                    g1ltd[i, k.item()] - gmax[
                                                                                        i, k.item()]) * (
                                                                                           1 - np.exp(vltd / 256))

                    if time == simulation_time:
                        for l in range(time - pulse_time_LTD, time):
                            if torch.numel(torch.nonzero(target_r[j])) != 0:
                                Ae_time_LTD = l  # Latest update time of LTD
                                Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                    -1)  # Latest update nueron index of LTD
                                if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                    X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                        -1)  # LTD causing spikes
                                    for i in range(X_size):
                                        if i in X_cause_index:
                                            X_cause_time = torch.nonzero(
                                                source_r[Ae_time_LTD:time, i]).reshape(-1)  # LTP causing spikes time
                                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                            for k in Ae_index_LTD:
                                                for j in range(X_cause_count):
                                                    t = pulse_time_LTD / abs(Ae_time_LTD - X_cause_time[j] + 1)
                                                    self.connection.w[i, k.item()] -= t * (self.connection.w[i, k.item()] +
                                                                                        g1ltd[i, k.item()] - gmax[
                                                                                            i, k.item()]) * (
                                                                                               1 - np.exp(vltd / 256))


        # Adapitve Drop Connect
        if ADC:
            p = np.round(minmax_scale(
                np.nan_to_num(self.connection.w.cpu().detach().numpy().reshape(X_size * Ae_size), copy=False),
                feature_range=(0.9995, 1)).reshape(X_size, Ae_size), 3)
            m = torch.zeros(X_size, Ae_size).to('cuda')

            for i in range(X_size):
                for j in range(Ae_size):
                    m[i, j] = int(bernoulli.rvs(p[i, j], size=1))

            self.connection.w *= m


        super().update()


class MemristiveSTDP_KIST(LearningRule):
    # language=rst
    """
    This rule is STDP with memristive characteristic.
    It involves both pre-synaptic and post-synaptic spiking activity.
    By default, pre-synaptic update is LTD and the post-synaptic update is LTP.
    In addtion, it updates the weight according to the time range between pre-synaptic and post-synaptic spikes.
    Input neurons' spiking proportion affects synaptic weight regulation.
    Also it is implified for KIST device.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MemristiveSTDP_Kist`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MemristiveSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self ,**kwargs) -> None:
        # language=rst
        """
        Memristive STDP learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Information of input and excitatory spikes
        source_s = self.source.s.view(-1).long()
        target_s = self.target.s.view(-1).long()
        X_size = torch.numel(source_s)
        Ae_size = torch.numel(target_s)

        # Spike recording variables
        s_record = kwargs.get('s_record', [])
        t_record = kwargs.get('t_record', [])
        simulation_time = kwargs.get('simulation_time')
        s_record.append(source_s.tolist())
        t_record.append(target_s.tolist())
        source_r = torch.tensor(s_record)
        target_r = torch.tensor(t_record)

        # Time variables
        time = len(source_r)
        ref_t = 22
        pulse_time_LTP = 50  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 50  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        X_cause_index = 0
        X_cause_count = 0
        X_cause_time = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Synaptic Template
        template_exc = []
        template_exc = kwargs.get('template_exc')
        ST = kwargs.get('ST')
        if ST:
            drop_index_input = kwargs.get('drop_index_input')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            for i in range(len(template_exc)):
                for j in drop_index_input[i]:
                    self.connection.w[j, template_exc[i]] = 0
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, template_exc[i]] <= 0.5:
                        self.connection.w[j, template_exc[i]] = 1.0


        # Dead synpase simulation
        dead_index_input = []
        dead_index_exc = []
        DS = kwargs.get('DS')
        if DS:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')
            for i in range(len(dead_index_exc)):
                for j in dead_index_input[i]:
                    self.connection.w[j, dead_index_exc[i]] = 0


        # Weight update with memristive characteristc
        if torch.numel(update_index_and_time) == 0:
            self.connection.w = self.connection.w

        elif torch.numel(update_index_and_time) != 0:
            if torch.numel(torch.nonzero(target_s)) != 0:
                Ae_time_LTP = time  # Latest update time
                Ae_index_LTP = torch.nonzero(target_s).view(-1)  # Latest update nueron index
            if Ae_time_LTP < pulse_time_LTP:
                if torch.sum(source_r[0:Ae_time_LTP]) > 0:  # LTP
                    X_cause_index = torch.nonzero(source_r[0:Ae_time_LTP])[:, [1]].view(
                        -1)  # LTP causing spikes
                    for i in range(X_size):
                        if i in X_cause_index:
                            X_cause_time = torch.nonzero(
                                source_r[0:Ae_time_LTP, i]).reshape(-1) # LTP causing spikes time
                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                            for k in Ae_index_LTP:
                                for j in range(X_cause_count):
                                    t = abs(Ae_time_LTP - X_cause_time[j])
                                    if (t <= ref_t):
                                        if (self.connection.w[i, k.item()] == 1.0):
                                            continue
                                        else:
                                            self.connection.w[i, k.item()] = 1.0

            elif Ae_time_LTP >= pulse_time_LTP:
                if torch.sum(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP]) > 0:  # LTP
                    X_cause_index = torch.nonzero(source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP])[:,
                                    [1]].view(
                        -1)  # LTP causing spikes
                    for i in range(X_size):
                        if i in X_cause_index:
                            X_cause_time = torch.nonzero(
                                source_r[Ae_time_LTP - pulse_time_LTP:Ae_time_LTP, i]).reshape(
                                -1) # LTP causing spikes time
                            X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                            for k in Ae_index_LTP:
                                for j in range(X_cause_count):
                                    t = abs(Ae_time_LTP - X_cause_time[j])
                                    if (t <= ref_t):
                                        if (self.connection.w[i, k.item()] == 1.0):
                                            continue
                                        else:
                                            self.connection.w[i, k.item()] = 1.0

                if time - pulse_time_LTD > 0:
                    if torch.numel(
                            torch.nonzero(target_r[time - pulse_time_LTD])) != 0:  # Checking LTD spike time
                        Ae_time_LTD = time - pulse_time_LTD  # Latest update time of LTD
                        Ae_index_LTD = torch.nonzero(
                            target_r[Ae_time_LTD])  # Latest update nueron index of LTD
                        if torch.sum(source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD]) > 0:  # LTD
                            X_cause_index = torch.nonzero(
                                source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD])[:, [1]].view(
                                -1)  # LTD causing spikes
                            for i in range(X_size):
                                if i in X_cause_index:
                                    X_cause_time = torch.nonzero(
                                        source_r[Ae_time_LTD:Ae_time_LTD + pulse_time_LTD, i]).reshape(
                                        -1)  # LTP causing spikes time
                                    X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                    for k in Ae_index_LTD:
                                        for j in range(X_cause_count):
                                            t = abs(Ae_time_LTP - X_cause_time[j])
                                            if (t <= ref_t):
                                                if (self.connection.w[i, k.item()] == 0.2):
                                                    continue
                                                else:
                                                    self.connection.w[i, k.item()] = 0.2

                if time == simulation_time:
                    for l in range(time - pulse_time_LTD, time):
                        if torch.numel(torch.nonzero(target_r[j])) != 0:
                            Ae_time_LTD = l  # Latest update time of LTD
                            Ae_index_LTD = torch.nonzero(target_r[j]).view(
                                -1)  # Latest update nueron index of LTD
                            if torch.sum(source_r[Ae_time_LTD:time]) > 0:  # LTD
                                X_cause_index = torch.nonzero(source_r[Ae_time_LTD:time])[:, [1]].view(
                                    -1)  # LTD causing spikes
                                for i in range(X_size):
                                    if i in X_cause_index:
                                        X_cause_time = torch.nonzero(
                                            source_r[Ae_time_LTD:time, i]).reshape(-1)  # LTP causing spikes time
                                        X_cause_count = torch.ne(X_cause_index, i).tolist().count(False)
                                        for k in Ae_index_LTD:
                                            for j in range(X_cause_count):
                                                t = abs(Ae_time_LTP - X_cause_time[j])
                                                if (t <= ref_t):
                                                    if (self.connection.w[i, k.item()] == 0.2):
                                                        continue
                                                    else:
                                                        self.connection.w[i, k.item()] = 0.2


        super().update()

class PostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        target_x = self.target.x.reshape(batch_size, out_channels * height_out, 1)
        target_x = target_x * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_x = target_x * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_x = target_x * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_x = (
            self.source.x.unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        dead_synapses = kwargs.get('dead_synapse')  # Dead synapses simulation

        # Dead synpase simulation
        if dead_synapses:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')

            for i in range(len(dead_index_exc)):
                for j in dead_index_input[i]:
                    self.connection.w[j, dead_index_exc[i]] = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any():
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(self.source.x, _pair(padding))
        source_x = source_x.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            # print(self.nu[0].shape, self.connection.w.size())
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv3dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(
            self.source.x,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_x = (
            source_x.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()
        # print(target_x.shape, source_s.shape, self.connection.w.shape)

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()
