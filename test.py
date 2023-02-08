import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Net, xydeg2xysincos, xysincos2xydeg
import prepare_dataset

net = Net()
net.load_state_dict(torch.load("trained.pth"))

test_data = prepare_dataset.make_random_data(1000)
test_labels = prepare_dataset.get_labels(test_data)

test_dataset = xydeg2xysincos(torch.from_numpy(test_data))
outputs = net(test_dataset)
outputs = xysincos2xydeg(outputs).detach().numpy()

place_errors = np.sqrt((test_labels[:, 0]-outputs[:, 0])**2+(test_labels[:, 1]-outputs[:, 1])**2)
angle_errors = np.minimum(np.abs(test_labels[:, 2]-outputs[:, 2]), 2*np.pi-np.abs(test_labels[:, 2]-outputs[:, 2]))
print("average position error:", np.mean(place_errors), "average angle error:", np.mean(angle_errors))
print("maximal position error:", np.max(place_errors), "maximal angle error:", np.max(angle_errors))

print(np.concatenate((test_labels[:5], outputs[:5]), 1))

plt.subplot(1, 2, 1)
for i in range(5):
    plt.arrow(test_data[i, 0], test_data[i, 1], np.cos(test_data[i, 2]), np.sin(test_data[i, 2]), width=0.1, length_includes_head=True, color="brown")
    plt.arrow(test_labels[i, 0], test_labels[i, 1], np.cos(test_labels[i, 2]), np.sin(test_labels[i, 2]), width=0.1, length_includes_head=True, color="gray")
    plt.arrow(outputs[i, 0], outputs[i, 1], np.cos(outputs[i, 2]), np.sin(outputs[i, 2]), width=0.06, length_includes_head=True, color="red")
plt.xlim((-1, 10))
plt.ylim((-1, 10))

plt.subplot(1, 2, 2)
plt.scatter(place_errors, angle_errors, s=0.5, c="darkblue")
plt.xlabel('chyba pozice')
plt.ylabel('chyba úhlu (rad)')

plt.show()
