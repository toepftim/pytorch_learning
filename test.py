import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Net
import prepare_dataset

net = Net()
net.load_state_dict(torch.load("trained.pth"))

test_data = prepare_dataset.make_random_data(1000)
test_labels = prepare_dataset.get_labels(test_data)

outputs = net(torch.from_numpy(test_data)).detach().numpy()

place_errors = np.sqrt((test_labels[:, 0]-outputs[:, 0])**2+(test_labels[:, 1]-outputs[:, 1])**2)
angle_errors = np.minimum(np.abs(test_labels[:, 2]-outputs[:, 2]), 2*np.pi-np.abs(test_labels[:, 2]-outputs[:, 2]))
print("average position error:", np.mean(place_errors), "average angle error:", np.mean(angle_errors))
print("maximal position error:", np.max(place_errors), "maximal angle error:", np.max(angle_errors))

print(outputs[0], test_labels[0])

fig, ax = plt.subplots()
for i in range(10):
    ax.arrow(test_data[i, 0], test_data[i, 1], np.cos(test_data[i, 2]), np.sin(test_data[i, 2]), width=0.1, length_includes_head=True, color="brown")
    ax.arrow(test_labels[i, 0], test_labels[i, 1], np.cos(test_labels[i, 2]), np.sin(test_labels[i, 2]), width=0.1, length_includes_head=True, color="gray")
    ax.arrow(outputs[i, 0], outputs[i, 1], np.cos(outputs[i, 2]), np.sin(outputs[i, 2]), width=0.1, length_includes_head=True, color="red")

ax.set(xlim=(-1, 10), ylim=(-1, 10))
plt.show()
