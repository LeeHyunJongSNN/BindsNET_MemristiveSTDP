import os
import torch
from bindsnet.network import Network
from bindsnet.datasets import FashionMNIST
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes

network = Network()

input_layer = Input(n=784, sum_input=True)
output_layer = LIFNodes(n=10, sum_input=True)
network.add_layer(input_layer, name="X")
network.add_layer(output_layer, name="Y")

input_connection = Connection(input_layer, output_layer, norm=150, wmin=-1, wmax=1)
network.add_connection(input_connection, source="X", target="Y")

time = 25
for layers in network.layers:
    m = Monitor(network.layers[layers], state_vars=['s'], time=time)
    network.add_monitor(m, name=layers)

raw = FashionMNIST(root=os.path.join("..", "..", "data", "FashionMNIST"), download=True, train=True)
images, labels = raw.data, raw.targets

grads = {}
lr, lr_decay = 1e-2, 0.95
criterion = torch.nn.CrossEntropyLoss()
spike_ims, spike_axes, Weights_im = None, None, None

for i, (image, label) in enumerate(zip(images.view(-1, 784) / 255, labels)):
    inputs = {"X": image.repeat(time, 1), 'Y_b': torch.ones(time, 1)}
    network.run(inputs=inputs, time=time)

    label = torch.tensor(label).long()
    spikes = {layers: network.monitors[layers].get("s") for layers in network.layers}
    summed_inputs = {layers: network.layers[layers].summed for layers in network.layers}

    output = network.monitors['Y'].get("s").sum(0).view(1, 10)
    predicted = output.argmax(1).item()

    grads['dl/df'] = summed_inputs['Y'].softmax(0)
    grads['dl/df'][label] -= 1
    grads['dl/df'] = torch.ger(summed_inputs['X'].squeeze(), grads['dl/df'])
    network.connections[('X', 'Y')].w -= lr * grads['dl/df']

    if i > 0 and i % 500 == 0:
        lr *= lr_decay

    print(predicted, label.item())

network.reset_()
