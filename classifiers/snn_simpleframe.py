import os
import gc
import torch
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time as t
from sklearn.model_selection import train_test_split

from bindsnet.encoding import PoissonEncoder, RankOrderEncoder, BernoulliEncoder, SingleEncoder, RepeatEncoder
from bindsnet.memstdp import RankOrderTTFSEncoder, RankOrderTTASEncoder
from bindsnet.memstdp.MemSTDP_models import AdaptiveIFNetwork_MemSTDP, DiehlAndCook2015_MemSTDP
from bindsnet.memstdp.MemSTDP_learning import MemristiveSTDP, MemristiveSTDP_Simplified, MemristiveSTDP_TimeProportion
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_assignments,
    plot_weights,
    plot_performance,
    plot_voltages,
)
from bindsnet.memstdp.plotting_weights_counts import hist_weights

random_seed = random.randint(0, 100)
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=random_seed)
parser.add_argument("--n_neurons", type=int, default=4)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=90)
parser.add_argument("--inh", type=float, default=480)
parser.add_argument("--rest", type=float, default=-65.0)
parser.add_argument("--reset", type=float, default=-60.0)
parser.add_argument("--thresh", type=float, default=-58.0)
parser.add_argument("--theta_plus", type=float, default=0.0009)
parser.add_argument("--time", type=int, default=500)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=12)      # increase by 2.5~4
parser.add_argument("--encoder_type", dest="encoder_type", default="PoissonEncoder")
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=100)
parser.add_argument("--test_ratio", type=float, default=0.95)
parser.add_argument("--random_G", type=bool, default=True)
parser.add_argument("--vLTP", type=float, default=0.0)
parser.add_argument("--vLTD", type=float, default=0.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(train_plot=False, test_plot=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_workers = args.n_workers
exc = args.exc
inh = args.inh
rest = args.rest
reset = args.reset
thresh = args.thresh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
encoder_type = args.encoder_type
progress_interval = args.progress_interval
update_interval = args.update_interval
test_ratio = args.test_ratio
random_G = args.random_G
vLTP = args.vLTP
vLTD = args.vLTD
beta = args.beta
train_plot = args.train_plot
test_plot = args.test_plot
gpu = args.gpu
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

print("Running on Device =", device)
print("Random Seed =", random_seed)
print("Random G value =", random_G)
print("vLTP =", vLTP)
print("vLTD =", vLTD)
print("beta =", beta)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

print(n_workers, os.cpu_count() - 1)

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

if encoder_type == "PoissonEncoder":
    encoder = PoissonEncoder(time=time, dt=dt)

elif encoder_type == "RankOrderEncoder":
    encoder = RankOrderEncoder(time=time, dt=dt)

elif encoder_type == "TTFSEncoder":
    encoder = RankOrderTTFSEncoder(time=time, dt=dt)

elif encoder_type == "TTASEncoder":
    encoder = RankOrderTTASEncoder(time=time, dt=dt)

elif encoder_type == "BernoulliEncoder":
    encoder = BernoulliEncoder(time=time, dt=dt)

elif encoder_type == "SingleEncoder":
    encoder = SingleEncoder(time=time, dt=dt)

elif encoder_type == "RepeatEncoder":
    encoder = RepeatEncoder(time=time, dt=dt)

else:
    print("Error!! There is no such encoder!!")


wave_data = []
classes = []

fname = "/home/leehyunjong/Wi-Fi_Preambles/"\
        "WIFI_10MHz_IQvector_(minus)3dB_20000.txt"

raw = np.loadtxt(fname, dtype='complex')

for line in raw:
    line_data = line[0:len(line) - 1]
    line_label = line[-1]
    fft1 = np.fft.fft(line_data[16:80])
    fft2 = np.fft.fft(line_data[96:160])
    fft3 = np.fft.fft(line_data[192:256])
    fft4 = np.fft.fft(line_data[256:len(line_data)])
    fft = np.concatenate((fft1, fft2, fft3, fft4), axis=0)
    scaled = intensity * np.abs(fft)

    if not gpu:
        if line_label == 0:
            line_label = 1
        else:
            line_label = 0

    classes.append(line_label)
    lbl = torch.tensor(line_label).long()

    converted = torch.tensor(scaled, dtype=torch.float32)
    encoded = encoder.enc(datum=converted, time=time, dt=dt)
    wave_data.append({"encoded_image": encoded, "label": lbl})

train_data, test_data = train_test_split(wave_data, test_size=test_ratio)

n_classes = (np.unique(classes)).size

n_train = len(train_data)
n_test = len(test_data)

num_inputs = train_data[-1]["encoded_image"].shape[1]

print(n_train, n_test, n_classes)

# Build network.
network = DiehlAndCook2015_MemSTDP(
    n_inpt=num_inputs,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    rest=rest,
    reset=reset,
    thresh=thresh,
    update_rule=MemristiveSTDP_Simplified,
    dt=dt,
    norm=1.0,
    theta_plus=theta_plus,
    inpt_shape=(1, num_inputs, 1),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=int(time / dt))
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=int(time / dt))
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
hist_ax = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Random variables
rand_gmax = torch.rand(num_inputs, n_neurons)
rand_gmin = rand_gmax / 10 + torch.rand(num_inputs, n_neurons) / 100

# Train the network.
print("\nBegin training.\n")
start = t()
print("check accuracy per", update_interval)
for epoch in range(n_epochs):
    labels = []
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    for step, batch in enumerate(tqdm(train_data)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, num_inputs, 1)}

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
                # Match a label of a neuron that has the proportion of highest spikes rate with a data's real label.
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
        network.run(inputs=inputs, time=time, input_time_dim=1, s_record=s_record, t_record=t_record,
                    simulation_time=time, rand_gmax=rand_gmax, rand_gmin=rand_gmin, random_G=random_G,
                    vLTP=vLTP, vLTD=vLTD, beta=beta, ST=False, DS=False, ADC=False)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if train_plot:
            image = batch["encoded_image"].view(num_inputs, time)
            inpt = inputs["X"].view(time, train_data[-1]["encoded_image"].shape[1]).sum(0).view(16, 16)
            input_exc_weights = network.connections[("X", "Ae")].w * 100    # Scaling for plotting
            square_weights = get_square_weights(
               input_exc_weights.view(train_data[-1]["encoded_image"].shape[1], n_neurons), n_sqrt, 16
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
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

            weight_collections = network.connections[("X", "Ae")].w.reshape(-1).tolist()
            hist_ax = hist_weights(weight_collections, ax=hist_ax)

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}
confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)

for step, batch in enumerate(test_data):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, num_inputs, 1)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    s_record = []
    t_record = []
    network.run(inputs=inputs, time=time, input_time_dim=1, s_record=s_record, t_record=t_record,
                simulation_time=time, rand_gmax=rand_gmax, rand_gmin=rand_gmin, random_G=random_G,
                vLTP=vLTP, vLTD=vLTD, beta=beta, ST=False, DS=False, ADC=False)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

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

    if test_plot:
        image = batch["encoded_image"].view(num_inputs, time)
        inpt = inputs["X"].view(time, test_data[-1]["encoded_image"].shape[1]).sum(0).view(16, 16)
        spikes_ = {layer: spikes[layer].get("s") for layer in spikes}

        spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
        inpt_axes, inpt_ims = plot_input(
            image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
        )

        plt.pause(1e-8)

    # print(accuracy["all"], label_tensor.long(), all_activity_pred)
    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

    if gpu:
        P_tensor = torch.zeros(label_tensor.long().shape).long().cuda()      # label 0
        N_tensor = torch.ones(label_tensor.long().shape).long().cuda()     # label 1

    else:
        P_tensor = torch.ones(label_tensor.long().shape).long()     # label 1
        N_tensor = torch.zeros(label_tensor.long().shape).long()    # label 0

    TP = float(torch.sum((all_activity_pred == P_tensor) & (label_tensor.long() == P_tensor)).item())
    FP = float(torch.sum((all_activity_pred == P_tensor) & (label_tensor.long() == N_tensor)).item())
    TN = float(torch.sum((all_activity_pred == N_tensor) & (label_tensor.long() == N_tensor)).item())
    FN = float(torch.sum((all_activity_pred == N_tensor) & (label_tensor.long() == P_tensor)).item())

    confusion_matrix["TP"] += TP
    confusion_matrix["FP"] += FP
    confusion_matrix["TN"] += TN
    confusion_matrix["FN"] += FN

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

    print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test * 100))
    print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test * 100))

print(confusion_matrix)
print("Probability of Detection: %.4f" % (confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])))
print("False Negative Probability: %.4f" % (confusion_matrix["FN"] / (confusion_matrix["TP"] + confusion_matrix["FN"])))
print("False Positive Probability: %.4f" % (confusion_matrix["FP"] / (confusion_matrix["FP"] + confusion_matrix["TN"])))
print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
