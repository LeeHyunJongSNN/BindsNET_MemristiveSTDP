import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

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

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean or string varibles for addtional feature
        g_rand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        ST = kwargs.get('ST')  # ST useage
        Pruning = kwargs.get('Pruning')  # Pruning useage
        FT = kwargs.get('fault_type')  # DS simulation

        # Random Conductance uperbound and underbound
        if g_rand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            n_neurons = kwargs.get('n_neurons')
            self.connection.w *= drop_mask
            for i in range(n_neurons):
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, i] <= gmax[j, i] * 0.4:  # min 0.4
                        self.connection.w[j, i] = gmax[j, i] * reinforce_ref[i][int(np.where(
                            j == reinforce_index_input[i])[0])] * 0.5  # scaling 0.5


        # Fault synapse application
        if FT == 'SA0':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= dead_mask

        elif FT == 'SA1':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= torch.where(dead_mask != 0, 0, torch.ones_like(dead_mask))
            self.connection.w += dead_mask


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


        # Network pruning
        if Pruning:
            check = torch.where(self.connection.w >= 0.1, True, False)
            self.connection.w *= check


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

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean or string varibles for addtional feature
        g_rand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        ST = kwargs.get('ST')  # ST useage
        Pruning = kwargs.get('Pruning')  # Pruning useage
        FT = kwargs.get('fault_type')  # DS simulation

        # Random Conductance uperbound and underbound
        if g_rand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            n_neurons = kwargs.get('n_neurons')
            self.connection.w *= drop_mask
            for i in range(n_neurons):
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, i] <= gmax[j, i] * 0.4:     # min 0.4
                        self.connection.w[j, i] = gmax[j, i] * reinforce_ref[i][int(np.where(
                            j == reinforce_index_input[i])[0])] * 0.5   # scaling 0.5


        # Fault synapse application
        if FT == 'SA0':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= dead_mask

        elif FT == 'SA1':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= torch.where(dead_mask != 0, 0, torch.ones_like(dead_mask))
            self.connection.w += dead_mask


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


        # Network pruning
        if Pruning:
            check = torch.where(self.connection.w >= 0.02, True, False)
            self.connection.w *= check


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

        # Random distribution Gmax and Gmin
        g_rand = kwargs.get('random_G')

        # Random Conductance uperbound and underbound
        if g_rand:
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

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean or string varibles for addtional feature
        g_rand = kwargs.get('random_G')  # Random distribution Gmax and Gmin
        ST = kwargs.get('ST')  # ST useage
        Pruning = kwargs.get("Pruning")  # Pruning useage
        FT = kwargs.get('fault_type')   # fault type

        # Random conductance uperbound and underbound
        if g_rand:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')
        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))


        # Synaptic Template
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            n_neurons = kwargs.get('n_neurons')
            self.connection.w *= drop_mask
            for i in range(n_neurons):
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, i] <= gmax[j, i] * 0.4:     # min 0.4
                        self.connection.w[j, i] = gmax[j, i] * reinforce_ref[i][int(np.where(
                            j == reinforce_index_input[i])[0])] * 0.5   # scaling 0.5


        # Fault synapse application
        if FT == 'SA0':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= dead_mask

        elif FT == 'SA1':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= torch.where(dead_mask != 0, 0, torch.ones_like(dead_mask))
            self.connection.w += dead_mask


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


        # Network pruning
        if Pruning:
            check = torch.where(self.connection.w >= 0.02, True, False)
            self.connection.w *= check


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
        high = kwargs.get('rand_gmax')
        low = kwargs.get('rand_gmin')
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
        ST = kwargs.get('ST')
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_index_input = kwargs.get('reinforce_index_input')
            reinforce_ref = kwargs.get('reinforce_ref')
            n_neurons = kwargs.get('n_neurons')
            self.connection.w *= drop_mask
            for i in range(n_neurons):
                for j in reinforce_index_input[i]:
                    if self.connection.w[j, i] < high[j, i]:
                        self.connection.w[j, i] = high[j, i]


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
                                        if (self.connection.w[i, k.item()] == high[i, k.item()]):
                                            continue
                                        else:
                                            self.connection.w[i, k.item()] = high[i, k.item()]

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
                                        if (self.connection.w[i, k.item()] == high[i, k.item()]):
                                            continue
                                        else:
                                            self.connection.w[i, k.item()] = high[i, k.item()]

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
                                                if (self.connection.w[i, k.item()] == low[i, k.item()]):
                                                    continue
                                                else:
                                                    self.connection.w[i, k.item()] = low[i, k.item()]

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
                                                    if (self.connection.w[i, k.item()] == low[i, k.item()]):
                                                        continue
                                                    else:
                                                        self.connection.w[i, k.item()] = low[i, k.item()]


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
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size


        # Additional functions
        ST = kwargs.get('ST')  # ST useage
        Pruning = kwargs.get('Pruning')  # Pruning useage
        FT = kwargs.get('fault_type')  # FS simulation


        # Synaptic Template
        # if ST:
        #     drop_mask = kwargs.get('drop_mask')
        #     reinforce_index_input = kwargs.get('reinforce_index_input')
        #     reinforce_ref = kwargs.get('reinforce_ref')
        #     n_neurons = kwargs.get('n_neurons')
        #     self.connection.w *= drop_mask
        #     for i in range(n_neurons):
        #         for j in reinforce_index_input[i]:
        #             if self.connection.w[j, i] <= 0.4:        # min 0.4
        #                 self.connection.w[j, i] = reinforce_ref[i][int(np.where(
        #                     j == reinforce_index_input[i])[0])] * 0.5      # scaling 0.5

        # Synaptic Template (vector version)
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_mask = kwargs.get('reinforce_mask')
            self.connection.w *= drop_mask
            self.connection.w *= torch.where((self.connection.w < 0.4) & (reinforce_mask != 0),     # min 0.4
                                                     0, torch.ones_like(self.connection.w))
            self.connection.w += reinforce_mask * 0.5  # scaling 0.5


        # Fault synapse application
        if FT == 'SA0':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= dead_mask

        elif FT == 'SA1':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= torch.where(dead_mask != 0, 0 ,torch.ones_like(dead_mask))
            self.connection.w += dead_mask


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


        # Network pruning
        if Pruning:
            check_pruning = torch.where(self.connection.w >= 0.02, True, False)
            self.connection.w *= check_pruning


        super().update()


class Thresh_PostPre(LearningRule):
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
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size


        # Additional functions
        ST = kwargs.get('ST')  # ST useage
        Pruning = kwargs.get('Pruning')  # Pruning useage
        FT = kwargs.get('fault_type')  # FS simulation


        # Synaptic Template
        # if ST:
        #     drop_mask = kwargs.get('drop_mask')
        #     reinforce_index_input = kwargs.get('reinforce_index_input')
        #     reinforce_ref = kwargs.get('reinforce_ref')
        #     n_neurons = kwargs.get('n_neurons')
        #     self.connection.w *= drop_mask
        #     for i in range(n_neurons):
        #         for j in reinforce_index_input[i]:
        #             if self.connection.w[j, i] <= 0.4:        # min 0.4
        #                 self.connection.w[j, i] = reinforce_ref[i][int(np.where(
        #                     j == reinforce_index_input[i])[0])] * 0.5      # scaling 0.5

        # Synaptic Template (vector version)
        if ST:
            drop_mask = kwargs.get('drop_mask')
            reinforce_mask = kwargs.get('reinforce_mask')
            self.connection.w *= drop_mask
            self.connection.w *= torch.where((self.connection.w < 0.5) & (reinforce_mask != 0),
                                             0, torch.ones_like(self.connection.w))
            self.connection.w += reinforce_mask * 0.4


        # Fault synapse application
        if FT == 'SA0':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= dead_mask

        elif FT == 'SA1':
            dead_mask = kwargs.get('dead_mask')
            self.connection.w *= torch.where(dead_mask != 0, 0, torch.ones_like(dead_mask))
            self.connection.w += dead_mask


        # Pre-synaptic update.
        if self.nu[0].any() and torch.sum(self.target.s.view(-1).long()) is not 0:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            delta_LTD = self.reduction(torch.bmm(source_s, target_x), dim=0)
            torch.where(self.connection.w < 0.1, 0, delta_LTD)
            self.connection.w -= delta_LTD
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any() and torch.sum(self.target.s.view(-1).long()) is not 0:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            delta_LTP = self.reduction(torch.bmm(source_x, target_s), dim=0)
            torch.where(self.connection.w > 0.4, 0, delta_LTP)
            self.connection.w += delta_LTP
            del source_x, target_s


        # Network pruning
        if Pruning:
            check_pruning = torch.where(self.connection.w >= 0.02, True, False)
            self.connection.w *= check_pruning


        super().update()


class STB_PostPre(LearningRule):
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
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size
        n_neurons = kwargs.get('n_neurons')
        num_inputs = kwargs.get('num_inputs')
        dt = kwargs.get('dt')
        offset = 0.1
        lam = 0.999
        scale = 24

        # Pre-synaptic update.
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            trace_LTD = lam * self.reduction(torch.bmm(source_s, target_x), dim=0)
            delta_LTD = scale * (trace_LTD - offset) / (batch_size * dt)
            self.connection.w -= delta_LTD
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any():
            target_s = (self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1])
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            trace_LTP = lam * self.reduction(torch.bmm(source_x, target_s), dim=0)
            delta_LTP = scale * (trace_LTP - offset) / (batch_size * dt)
            self.connection.w += delta_LTP
            del source_x, target_s

        super().update()

