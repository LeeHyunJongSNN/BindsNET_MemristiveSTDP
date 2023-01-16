import argparse
import os
import gc
from time import time as t

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.signal import detrend

from bindsnet.encoding import PoissonEncoder
from bindsnet.memstdp.MemSTDP_models import DiehlAndCook2015_MemSTDP
from bindsnet.memstdp.MemSTDP_learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection

from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_assignments,
    plot_weights,
    plot_performance,
    plot_voltages,
)
from bindsnet.memstdp.plotting_weights_counts import hist_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--encoding_time", type=int, default=500)
parser.add_argument("--encoding_dt", type=int, default=1.0)
parser.add_argument("--n_neurons", type=int, default=4)
parser.add_argument("--exc", type=int, default=22.5)
parser.add_argument("--inh", type=int, default=17.5)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--intensity", type=float, default=800.0)
parser.add_argument("--scale", type=float, default=600.0)
parser.add_argument("--dropout_num", type=int, default=20)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=10)
parser.add_argument("--test_ratio", type=float, default=0.999)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.add_argument("--dropout", type=bool, default=True)
parser.set_defaults(conv_plot=False, asd_plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_workers = args.n_workers
batch_size = args.batch_size
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
encoding_time = args.encoding_time
encoding_dt = args.encoding_dt
n_neurons = args.n_neurons
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
intensity = args.intensity
scale = args.scale
dropout_num = args.dropout_num
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
test_ratio = args.test_ratio
conv_plot = args.plot
asd_plot = args.plot
gpu = args.gpu
dropout = args.dropout
spare_gpu = args.spare_gpu

# Sets up Gpu use
gc.collect()
torch.cuda.empty_cache()

if spare_gpu != 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spare_gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)

print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

print(n_workers, os.cpu_count() - 1)

train_data = []
test_data = []

wave_train_data = []
classes = []

conv_encoder = PoissonEncoder(time=time, dt=dt)

fname = "D:/SNN_dataset/Wi-Fi_Preambles/"\
        "WIFI_10MHz_IQvector_18dB_20000.txt"

raw = np.loadtxt(fname, dtype='complex')

for line in raw:
    line_data = line[0:len(line) - 1]
    line_label = line[-1]
    dcr = detrend(line_data - np.mean(line_data))
    fft1 = np.fft.fft(dcr[16:80]) / 64
    fft2 = np.fft.fft(dcr[96:160]) / 64
    fft3 = np.fft.fft(dcr[192:256]) / 64
    fft4 = np.fft.fft(dcr[256:len(dcr)]) / 64
    fft = np.concatenate((fft1, fft2, fft3, fft4), axis=0)
    scaled = intensity * np.abs(fft)

    classes.append(line_label)
    lbl = torch.tensor(line_label).long()

    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = conv_encoder.enc(datum=converted, time=time, dt=dt)
    wave_train_data.append({"encoded_image": encoded, "label": lbl})

train_data, test_data = train_test_split(wave_train_data, test_size=test_ratio)

num_inputs = train_data[-1]["encoded_image"].shape[1]
input_size = int(np.sqrt(num_inputs))

conv_size = int((input_size - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

n_train = len(train_data)
n_test = len(test_data)

# Build network.
network_conv = Network()
input_layer = Input(n=256, shape=(1, input_size, input_size), traces=True)

n_nodes = n_filters * conv_size * conv_size
conv_layer = DiehlAndCookNodes(
    n=n_nodes,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size**2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network_conv.add_layer(input_layer, name="X")
network_conv.add_layer(conv_layer, name="Y")
network_conv.add_connection(conv_conn, source="X", target="Y")
network_conv.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network_conv.layers["Y"], ["v"], time=time)
network_conv.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network_conv.to("cuda")

conv_spikes = {}
for layer in set(network_conv.layers):
    conv_spikes[layer] = Monitor(network_conv.layers[layer], state_vars=["s"], time=time)
    network_conv.add_monitor(conv_spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network_conv.layers) - {"X"}:
    voltages[layer] = Monitor(network_conv.layers[layer], state_vars=["v"], time=time)
    network_conv.add_monitor(voltages[layer], name="%s_voltages" % layer)

conv_train_data = []
conv_train_labels = []

# Train the network.
print("Begin Conv training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, batch in enumerate(tqdm(train_data)):
        # Get next input sample.
        if step > n_test:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, input_size, input_size)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network_conv.run(inputs=inputs, time=time)

        conv_train_data.append(torch.sum(conv_spikes["Y"].get("s").view(time, -1), 1).tolist())
        conv_train_labels.append([label.tolist()])

        # Optionally plot various simulation information.
        if conv_plot and batch_size == 1:
            image = batch["encoded_image"].view(num_inputs, time)

            inpt = inputs["X"].view(time, num_inputs).sum(0).view(input_size, input_size)
            weights1 = conv_conn.w
            _spikes = {
                "X": conv_spikes["X"].get("s").view(time, -1),
                "Y": conv_spikes["Y"].get("s").view(time, -1),
            }
            _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
            voltage_ims, voltage_axes = plot_voltages(
                _voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1)

        network_conv.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Conv Training complete.\n")

wave_train_data = []
dropout_index = []
pre_average = []
conv_classes = []

conv_train_data = np.array(conv_train_data)
conv_train_labels = np.array(conv_train_labels)
whole_data = np.concatenate((conv_train_data, conv_train_labels), axis=1)

asd_encoder = PoissonEncoder(time=encoding_time, dt=encoding_dt)

for line in whole_data:
    data = line[0:len(line) - 1]
    label = line[-1]
    scaled = scale * data
    conv_classes.append(label)
    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = asd_encoder.enc(datum=converted, time=encoding_time, dt=encoding_dt)
    wave_train_data.append({"encoded_image": encoded, "label": label})

num_inputs = wave_train_data[-1]["encoded_image"].shape[1]
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

# Build network.
network_asd = DiehlAndCook2015_MemSTDP(
    n_inpt=num_inputs,
    n_neurons=n_neurons,
    update_rule=PostPre,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=1.0,
    theta_plus=theta_plus,
    inpt_shape=(1, num_inputs, 1),
)

preprocessed = np.concatenate((conv_train_data, conv_train_labels), axis=1)
preprocessed = preprocessed[preprocessed[:, time].argsort()]
preprocessed = preprocessed[:, 0:time]

n_classes = (np.unique(conv_classes)).size
pre_size = int(np.shape(preprocessed)[0] / n_classes)

n_train = len(train_data)
n_test = len(test_data)

for j in range(n_classes):
    pre_average.append(np.mean(preprocessed[j * pre_size:(j + 1) * pre_size], axis=0))
    dropout_index.append(np.argwhere(pre_average[j] < np.sort(pre_average[j])[0:dropout_num + 1][-1]).flatten())

dropout_index *= int(np.ceil(n_neurons / n_classes))
dropout_exc = np.arange(n_neurons)

# Directs network to GPU
if gpu:
    network_asd.to("cuda")

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(encoding_time / encoding_dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network_asd.layers["Ae"], ["v"], time=int(encoding_time / dt))
inh_voltage_monitor = Monitor(network_asd.layers["Ai"], ["v"], time=int(encoding_time / dt))
network_asd.add_monitor(exc_voltage_monitor, name="exc_voltage")
network_asd.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
asd_spikes = {}
for layer in set(network_asd.layers):
    asd_spikes[layer] = Monitor(
        network_asd.layers[layer], state_vars=["s"], time=int(encoding_time / dt), device=device
    )
    network_asd.add_monitor(asd_spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network_asd.layers) - {"X"}:
    voltages[layer] = Monitor(
        network_asd.layers[layer], state_vars=["v"], time=int(encoding_time / dt), device=device
    )
    network_asd.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
hist_ax = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin ASD training.\n")
start = t()
print("check accuracy per", update_interval)
for epoch in range(n_epochs):
    labels = []
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, batch in enumerate(tqdm(wave_train_data)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(encoding_time / encoding_dt), 1, 1, num_inputs, 1)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)
            # Get network predictions.

            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=assignments,
                n_labels=n_classes,
            )

            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
                # Match a label of a neuron that has the highest rate of spikes with a data's real label.
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
                # Match a label of a neuron that has the proportion of the highest spike rate with a data's real label.
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])

        # Run the network on the input.
        s_record = []
        t_record = []
        network_asd.run(inputs=inputs, time=encoding_time, input_time_dim=1,
                        dead_synapse=dropout, dead_index_input=dropout_index, dead_index_exc=dropout_exc)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = asd_spikes["Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if asd_plot:
            image = batch["encoded_image"].view(num_inputs, encoding_time)
            inpt = inputs["X"].view(encoding_time, wave_train_data[-1]["encoded_image"].shape[1]).sum(0).view(1, num_inputs)
            input_exc_weights = network_asd.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(wave_train_data[-1]["encoded_image"].shape[1], n_neurons), n_sqrt, (1, num_inputs)
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: asd_spikes[layer].get("s") for layer in asd_spikes}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            weight_collections = network_asd.connections[("X", "Ae")].w.reshape(-1).tolist()
            hist_ax = hist_weights(weight_collections, ax=hist_ax)

            plt.pause(1e-8)

        network_asd.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("ASD Training complete.\n")

conv_test_data = []
conv_test_labels = []

print("\nBegin Conv testing\n")
network_conv.train(mode=False)
start = t()

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, batch in enumerate(tqdm(test_data)):
        # Get next input sample.
        if step > n_test:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, input_size, input_size)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network_conv.run(inputs=inputs, time=time)

        conv_test_data.append(torch.sum(conv_spikes["Y"].get("s").view(time, -1), 1).tolist())
        conv_test_labels.append([label.tolist()])

        network_conv.reset_state_variables()  # Reset state variables.

conv_test_data = np.array(conv_test_data)
conv_test_labels = np.array(conv_test_labels)
whole_data = np.concatenate((conv_test_data, conv_test_labels), axis=1)
wave_test_data = []

for line in whole_data:
    data = line[0:len(line) - 1]
    label = line[-1]
    scaled = scale * data
    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = asd_encoder.enc(datum=converted, time=encoding_time, dt=encoding_dt)
    wave_test_data.append({"encoded_image": encoded, "label": label})

wave_test_data = np.array(wave_test_data)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(encoding_time / encoding_dt), n_neurons), device=device)

# Train the network.
print("\nBegin ASD testing\n")
network_asd.train(mode=False)
start = t()

pbar = tqdm(total=n_test)

for step, batch in enumerate(wave_test_data):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(encoding_time / encoding_dt), 1, 1, num_inputs, 1)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    s_record = []
    t_record = []
    network_asd.run(inputs=inputs, time=encoding_time, input_time_dim=1,
                    dead_synapse=dropout, dead_index_input=dropout_index, dead_index_exc=dropout_exc)

    # Add to spikes recording.
    spike_record[0] = asd_spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record,
        assignments=assignments,
        n_labels=n_classes
    )

    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # print(accuracy["all"], label_tensor.long(), all_activity_pred)
    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

    network_asd.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

    print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test * 100))
    print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test * 100))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
