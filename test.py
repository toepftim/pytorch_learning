import torch
import numpy as np

from model import Net
import prepare_dataset

net = Net()
net.load_state_dict(torch.load("trained.pth"))

test_data = prepare_dataset.make_random_data(100)
test_labels = prepare_dataset.get_labels(test_data)

outputs = net(torch.from_numpy(test_data))

place_errors = np.sqrt((test_labels[:, 0]-outputs[:, 0].detach().numpy())**2+(test_labels[:, 1]-outputs[:, 1].detach().numpy())**2)
angle_errors = np.abs(test_labels[:, 2]-outputs[:, 2].detach().numpy())
print("average position error:", np.mean(place_errors), "average angle error:", np.mean(angle_errors))

