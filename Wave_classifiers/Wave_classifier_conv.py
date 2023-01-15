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

from bindsnet.analysis.plotting import (
    plot_conv2d_weights,
    plot_input,
    plot_spikes,
    plot_voltages,
)

from bindsnet.encoding import PoissonEncoder, RankOrderEncoder, BernoulliEncoder, SingleEncoder, RepeatEncoder
from bindsnet.memstdp import RankOrderTTFSEncoder
from bindsnet.memstdp.MemSTDP_models import AdaptiveIFNetwork_MemSTDP, DiehlAndCook2015_MemSTDP
from bindsnet.memstdp.MemSTDP_learning import MemristiveSTDP, MemristiveSTDP_Simplified, MemristiveSTDP_TimeProportion
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection

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
parser.add_argument("--intensity", type=float, default=1500)
parser.add_argument("--norm", type=float, default=16.0)
parser.add_argument("--encoder_type", dest="encoder_type", default="PoissonEncoder")
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--test_ratio", type=float, default=0.5)
parser.add_argument("--random_G", type=bool, default=True)
parser.add_argument("--vLTP", type=float, default=0.0)
parser.add_argument("--vLTD", type=float, default=0.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(plot=True, gpu=True, train=True)

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
random_G = args.random_G
vLTP = args.vLTP
vLTD = args.vLTD
beta = args.beta
enocder_type = args.encoder_type
intensity = args.intensity
norm = args.norm
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
test_ratio = args.test_ratio
plot = args.plot
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

print("Running on Device = ", device)
print("Random G value =", random_G)
print("vLTP =", vLTP)
print("vLTD =", vLTD)
print("beta =", beta)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

print(n_workers, os.cpu_count() - 1)

if enocder_type == "PoissonEncoder":
    encoder = PoissonEncoder(time=time, dt=dt)

elif enocder_type == "RankOrderEncoder":
    encoder = RankOrderEncoder(time=time, dt=dt)

elif enocder_type == "RankOrderTTFSEncoder":
    encoder = RankOrderTTFSEncoder(time=time, dt=dt)

elif enocder_type == "BernoulliEncoder":
    encoder = BernoulliEncoder(time=time, dt=dt)

elif enocder_type == "RepeatEncoder":
    encoder = RepeatEncoder(time=time, dt=dt)

else:
    print("Error!! There is no such encoder!!")

train_data = []
test_data = []

wave_data = []
classes = []

fname = "/home/leehyunjong/Wi-Fi_Preambles/"\
        "WIFI_10MHz_IQvector_18dB_60000.txt"

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
    encoded = encoder.enc(datum=converted, time=time, dt=dt)
    wave_data.append({"encoded_image": encoded, "label": lbl})

train_data, test_data = train_test_split(wave_data, test_size=test_ratio)

num_inputs = train_data[-1]["encoded_image"].shape[1]
input_size = int(np.sqrt(num_inputs))

conv_size = int((input_size - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=256, shape=(1, input_size, input_size), traces=True)

n_nodes = n_filters * conv_size * conv_size
conv_layer = DiehlAndCookNodes(
    n=n_nodes,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

rand_gmax = 0.5 * torch.rand(n_filters, 1, kernel_size, kernel_size).to(device) + 0.5
rand_gmin = 0.5 * torch.rand(n_filters, 1, kernel_size, kernel_size).to(device)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=MemristiveSTDP,
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

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
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
        if step > len(test_data):
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, input_size, input_size)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time, norm=norm,
                    rand_gmax=rand_gmax, rand_gmin=rand_gmin, random_G=random_G,
                    vLTP=vLTP, vLTD=vLTD, beta=beta)

        # Optionally plot various simulation information.
        if plot and batch_size == 1:
            image = batch["encoded_image"].view(num_inputs, time)

            inpt = inputs["X"].view(time, num_inputs).sum(0).view(input_size, input_size)
            weights1 = conv_conn.w
            _spikes = {
                "X": spikes["X"].get("s").view(time, -1),
                "Y": spikes["Y"].get("s").view(time, -1),
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

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
