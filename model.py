import torch
import torch.nn.functional as F
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 3)

    def forward(self, x):
        angles = x[:, 2:3]
        sins = np.sin(angles.numpy())
        coses = np.cos(angles.numpy())
        x = torch.concat((x, torch.from_numpy(sins), torch.from_numpy(coses)),1)
        x = self.fc1(x)
        return x
