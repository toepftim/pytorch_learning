import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from visualize import visualize_prediction
from model import Net, xyrad2xysincos, xysincos2xyrad
import prepare_dataset

net = Net()
net.load_state_dict(torch.load("trained.pth"))

test_data = prepare_dataset.make_grid_data()
# test_data = np.tile(np.array([[2.5, 0.5, 0], [0.5, 5.5, 0], [5.5, 6.5, 0]], np.float32), (100, 1))
test_data, test_labels = prepare_dataset.get_data_and_labels(test_data)

test_dataset = xyrad2xysincos(torch.from_numpy(test_data))
test_dataset = torch.tile(test_dataset, (10, 1))
test_labels = np.tile(test_labels, (10, 1))
test_dataset = torch.cat((test_dataset, 4*torch.rand(test_dataset.size()[0], 1)), 1)
outputs = net(test_dataset)
outputs = xysincos2xyrad(outputs).detach().numpy()

place_errors = np.sqrt((test_labels[:, 0]-outputs[:, 0])**2+(test_labels[:, 1]-outputs[:, 1])**2)
angle_errors = np.minimum(np.abs(test_labels[:, 2]-outputs[:, 2]), 2*np.pi-np.abs(test_labels[:, 2]-outputs[:, 2]))
print("average position error:", np.mean(place_errors), "average angle error:", np.mean(angle_errors))
print("maximal position error:", np.max(place_errors), "maximal angle error:", np.max(angle_errors))

figs, axes = plt.subplots(1, 2)
visualize_idxs = random.choices(range(len(test_data)), k=50)
visualize_prediction(axes[0], test_data[visualize_idxs], test_labels[visualize_idxs], outputs[visualize_idxs])
axes[0].set_xlim((0, 7))
axes[0].set_ylim((0, 7))


axes[1].scatter(place_errors, angle_errors, s=0.5, c="darkblue")
axes[1].set_xlabel('chyba pozice')
axes[1].set_ylabel('chyba úhlu (rad)')
axes[1].set_title(f"{len(outputs)} vzorků")

plt.show()
