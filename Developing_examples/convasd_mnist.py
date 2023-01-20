import argparse
import os
from time import time as t
import numpy as np

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.memstdp.MemSTDP_learning import PostPre
from bindsnet.memstdp.MemSTDP_models import DiehlAndCook2015_MemSTDP
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
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=400)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--conv_time", type=int, default=25)
parser.add_argument("--conv_dt", type=int, default=1.0)
parser.add_argument("--asd_time", type=int, default=250)
parser.add_argument("--asd_dt", type=int, default=1.0)
parser.add_argument("--n_neurons", type=int, default=10)
parser.add_argument("--exc", type=int, default=22.5)
parser.add_argument("--inh", type=int, default=17.5)
parser.add_argument("--theta_plus", type=float, default=0.02)
parser.add_argument("--conv_scale", type=float, default=512.0)
parser.add_argument("--asd_scale", type=float, default=10000.0)
parser.add_argument("--dropout_num", type=int, default=15)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--dropout", type=bool, default=True)
parser.set_defaults(conv_plot=False, asd_plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
batch_size = args.batch_size
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
conv_time = args.conv_time
conv_dt = args.conv_dt
asd_time = args.asd_time
asd_dt = args.asd_dt
n_neurons = args.n_neurons
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
conv_scale = args.conv_scale
asd_scale = args.asd_scale
dropout_num = args.dropout_num
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
conv_plot = args.conv_plot
asd_plot = args.asd_plot
dropout = args.dropout
gpu = args.gpu

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

if not train:
    update_interval = n_test

conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network_conv = Network()
input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
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
voltage_monitor = Monitor(network_conv.layers["Y"], ["v"], time=conv_time)
network_conv.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network_conv.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=conv_time, dt=conv_dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * conv_scale)]
    ),
)

conv_spikes = {}
for layer in set(network_conv.layers):
    conv_spikes[layer] = Monitor(network_conv.layers[layer], state_vars=["s"], time=conv_time)
    network_conv.add_monitor(conv_spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network_conv.layers) - {"X"}:
    voltages[layer] = Monitor(network_conv.layers[layer], state_vars=["v"], time=conv_time)
    network_conv.add_monitor(voltages[layer], name="%s_voltages" % layer)

conv_train_data = []
conv_train_labels = []
pca = PCA(1)
scaler = MinMaxScaler()

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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=gpu,
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.
        if step > n_train:
            break
        inputs = {"X": batch["encoded_image"].view(conv_time, batch_size, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network_conv.run(inputs=inputs, time=conv_time)

        org = conv_spikes["Y"].get("s").view(conv_time, -1).int().numpy()
        pca.fit(org)
        dec = pca.transform(org).squeeze()
        conv_train_data.append(np.abs(dec).tolist())
        conv_train_labels.append(label.tolist())

        # Optionally plot various simulation information.
        if conv_plot and batch_size == 1:
            image = batch["image"].view(28, 28)

            inpt = inputs["X"].view(conv_time, 784).sum(0).view(28, 28)
            weights1 = conv_conn.w
            _spikes = {
                "X": conv_spikes["X"].get("s").view(conv_time, -1),
                "Y": conv_spikes["Y"].get("s").view(conv_time, -1),
            }
            _voltages = {"Y": voltages["Y"].get("v").view(conv_time, -1)}

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

wave_data = []
dropout_index = []
pre_average = []
conv_classes = []
encoder = PoissonEncoder(time=asd_time, dt=asd_dt)

conv_train_data = np.array(conv_train_data)
conv_train_labels = np.array(conv_train_labels)
whole_data = np.concatenate((conv_train_data, conv_train_labels), axis=1)

for line in whole_data:
    data = line[0:len(line) - 1]
    label = line[-1]
    scaler.fit(data.reshape(-1, 1))
    data = scaler.transform(data.reshape(-1, 1)).squeeze()
    scaled = asd_scale * np.round(data, 5)
    conv_classes.append(label)
    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = encoder.enc(datum=converted, time=asd_time, dt=asd_dt)
    wave_data.append({"encoded_image": encoded, "label": label})

num_inputs = wave_data[-1]["encoded_image"].shape[1]
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

# Build network.
network_asd = DiehlAndCook2015_MemSTDP(
    n_inpt=num_inputs,
    n_neurons=n_neurons,
    update_rule=PostPre,
    exc=exc,
    inh=inh,
    dt=conv_dt,
    norm=1.0,
    theta_plus=theta_plus,
    inpt_shape=(1, num_inputs, 1),
)

preprocessed = whole_data[whole_data[:, conv_time].argsort()]
preprocessed = preprocessed[:, 0:conv_time]

n_classes = (np.unique(conv_classes)).size
pre_size = int(np.shape(preprocessed)[0] / n_classes)

for j in range(10):
    pre_average.append(np.mean(preprocessed[j * pre_size:(j + 1) * pre_size], axis=0))
    dropout_index.append(np.argwhere(pre_average[j] < np.sort(pre_average[j])[0:dropout_num + 1][-1]).flatten())

dropout_index *= int(np.ceil(n_neurons / n_classes))
dropout_exc = np.arange(n_neurons)

# Directs network to GPU
if gpu:
    network_asd.to("cuda")

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(asd_time / asd_dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network_asd.layers["Ae"], ["v"], time=int(asd_time / conv_dt))
inh_voltage_monitor = Monitor(network_asd.layers["Ai"], ["v"], time=int(asd_time / conv_dt))
network_asd.add_monitor(exc_voltage_monitor, name="exc_voltage")
network_asd.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
asd_spikes = {}
for layer in set(network_asd.layers):
    asd_spikes[layer] = Monitor(
        network_asd.layers[layer], state_vars=["s"], time=int(asd_time / conv_dt), device=device
    )
    network_asd.add_monitor(asd_spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network_asd.layers) - {"X"}:
    voltages[layer] = Monitor(
        network_asd.layers[layer], state_vars=["v"], time=int(asd_time / conv_dt), device=device
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

    for step, batch in enumerate(tqdm(wave_data)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(asd_time / asd_dt), 1, 1, num_inputs, 1)}
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
        network_asd.run(inputs=inputs, time=asd_time, input_time_dim=1,
                        dead_synapse=dropout, dead_index_input=dropout_index, dead_index_exc=dropout_exc)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = asd_spikes["Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if asd_plot:
            image = batch["encoded_image"].view(num_inputs, asd_time)
            inpt = inputs["X"].view(asd_time, wave_data[-1]["encoded_image"].shape[1]).sum(0).view(1, num_inputs)
            input_exc_weights = network_asd.connections[("X", "Ae")].w * conv_time / 10
            square_weights = get_square_weights(
                input_exc_weights.view(wave_data[-1]["encoded_image"].shape[1], n_neurons), n_sqrt, (1, num_inputs)
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

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=conv_time, dt=conv_dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * conv_scale)]
    ),
)

conv_test_data = []
conv_test_labels = []

print("\nBegin Conv testing\n")
network_conv.train(mode=False)
start = t()

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=gpu,
    )

    for step, batch in enumerate(tqdm(test_dataloader)):
        # Get next input sample.
        if step > n_test:
            break
        inputs = {"X": batch["encoded_image"].view(conv_time, batch_size, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network_conv.run(inputs=inputs, time=conv_time)

        org = conv_spikes["Y"].get("s").view(conv_time, -1).int().numpy()
        pca.fit(org)
        dec = pca.transform(org).squeeze()
        conv_test_data.append(np.abs(dec).tolist())
        conv_test_labels.append(label.tolist())

        network_conv.reset_state_variables()  # Reset state variables.

conv_test_data = np.array(conv_test_data)
conv_test_labels = np.array(conv_test_labels)
whole_data = np.concatenate((conv_test_data, conv_test_labels), axis=1)
test_data = []

for line in whole_data:
    data = line[0:len(line) - 1]
    label = line[-1]
    scaler.fit(data.reshape(-1, 1))
    data = scaler.transform(data.reshape(-1, 1)).squeeze()
    scaled = asd_scale * np.round(data, 5)
    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = encoder.enc(datum=converted, time=asd_time, dt=asd_dt)
    test_data.append({"encoded_image": encoded, "label": label})

test_data = np.array(test_data)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(asd_time / asd_dt), n_neurons), device=device)

# Train the network.
print("\nBegin ASD testing\n")
network_asd.train(mode=False)
start = t()

pbar = tqdm(total=n_test)

for step, batch in enumerate(test_data):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(asd_time / asd_dt), 1, 1, num_inputs, 1)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network_asd.run(inputs=inputs, time=asd_time, input_time_dim=1,
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