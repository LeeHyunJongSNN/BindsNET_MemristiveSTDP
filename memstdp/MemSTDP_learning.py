import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
import torch

from bindsnet.utils import im2col_indices

from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
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
            ``NonLinear`` learning rule will modify.
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
        time = len(source_r) - 1
        pulse_time_LTP = 45  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 45   # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        update_time_index = 0
        update_num_index = 0
        X_cause_index = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Dead synapses variables
        dead_index_input = []
        dead_index_exc = []
        dead_synapse_input_num = kwargs.get('dead_synapse_input_num')
        dead_synapse_exc_num = kwargs.get('dead_synapse_exc_num')

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get("random_G")  # Random distribution Gmax and Gmin
        dead_synapses = kwargs.get('dead_synapse')  # Dead synapses simulation

        # Random Conductance uperbound and underbound
        if grand == True:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')

        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))

        # Dead synpase simulation
        if dead_synapses == True:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')

            for i in range(dead_synapse_input_num):
                for l in range(dead_synapse_exc_num):
                    self.connection.w[dead_index_input[i], dead_index_exc[l]] = 0

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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

        super().update()


    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Simplified Memristive STDP learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
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
        time = len(source_r) - 1
        pulse_time_LTP = 45  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 45  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        update_time_index = 0
        update_num_index = 0
        X_cause_index = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Dead synapses variables
        dead_index_input = []
        dead_index_exc = []
        dead_synapse_input_num = kwargs.get('dead_synapse_input_num')
        dead_synapse_exc_num = kwargs.get('dead_synapse_exc_num')

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get("random_G")  # Random distribution Gmax and Gmin
        dead_synapses = kwargs.get('dead_synapse')  # Dead synapses simulation

        # Random Conductance uperbound and underbound
        if grand == True:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')

        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))

        # Dead synpase simulation
        if dead_synapses == True:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')

            for i in range(dead_synapse_input_num):
                for l in range(dead_synapse_exc_num):
                    self.connection.w[dead_index_input[i], dead_index_exc[l]] = 0

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
                                            self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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
                                            self.connection.w[i, k.item()] -= (gmax[i, k.item()] -
                                                                               gmin[i, k.item()]) / 256

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )

            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

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
            ``NonLinear`` learning rule will modify.
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
        time = len(source_r) - 1
        pulse_time_LTP = 45  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 45   # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        update_time_index = 0
        update_num_index = 0
        X_cause_index = 0
        X_cause_count = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Dead synapses variables
        dead_index_input = []
        dead_index_exc = []
        dead_synapse_input_num = kwargs.get('dead_synapse_input_num')
        dead_synapse_exc_num = kwargs.get('dead_synapse_exc_num')

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get("random_G")  # Random distribution Gmax and Gmin
        dead_synapses = kwargs.get('dead_synapse')  # Dead synapses simulation

        # Random Conductance uperbound and underbound
        if grand == True:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')

        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))

        # Dead synpase simulation
        if dead_synapses == True:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')

            for i in range(dead_synapse_input_num):
                for j in range(dead_synapse_exc_num):
                    self.connection.w[dead_index_input[i], dead_index_exc[j]] = 0

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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

        super().update()


    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Memristive learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
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
        time = len(source_r) - 1
        pulse_time_LTP = 45  # Change this factcor when you want to change LTP time slot
        pulse_time_LTD = 45  # Change this factcor when you want to change LTD time slot

        # STDP time record variables
        update_index_and_time = torch.nonzero(target_r)
        update_time_index = 0
        update_num_index = 0
        X_cause_index = 0
        X_cause_count = 0
        Ae_time_LTP = 0
        Ae_time_LTD = 0
        Ae_index_LTP = 0
        Ae_index_LTD = 0

        # Dead synapses variables
        dead_index_input = []
        dead_index_exc = []
        dead_synapse_input_num = kwargs.get('dead_synapse_input_num')
        dead_synapse_exc_num = kwargs.get('dead_synapse_exc_num')

        # Factors for nonlinear update
        vltp = kwargs.get('vLTP')
        vltd = kwargs.get('vLTD')
        b = kwargs.get('beta')
        gmax = torch.zeros_like(self.connection.w) + 1
        gmin = torch.zeros_like(self.connection.w)

        # Boolean varibles for addtional feature
        grand = kwargs.get("random_G")  # Random distribution Gmax and Gmin
        dead_synapses = kwargs.get('dead_synapse')  # Dead synapses simulation

        # Random Conductance uperbound and underbound
        if grand == True:
            gmax = kwargs.get('rand_gmax')
            gmin = kwargs.get('rand_gmin')

        g1ltp = (gmax - gmin) / (1.0 - np.exp(-vltp))
        g1ltd = (gmax - gmin) / (1.0 - np.exp(-vltd))

        # Dead synpase simulation
        if dead_synapses == True:
            dead_index_input = kwargs.get('dead_index_input')
            dead_index_exc = kwargs.get('dead_index_exc')

            for i in range(dead_synapse_input_num):
                for j in range(dead_synapse_exc_num):
                    self.connection.w[dead_index_input[i], dead_index_exc[j]] = 0

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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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

                    if time == simulation_time - 1:
                        for l in range(time - pulse_time_LTD, time + 1):
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
        if self.nu[0]:
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )

            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()
