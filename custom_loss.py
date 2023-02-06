import torch
import numpy as np


def custom_loss(outputs: torch.Tensor, labels: torch.Tensor):
    position_losses = (outputs[:, 0]-labels[:, 0])**2+(outputs[:, 1]-labels[:, 1])**2
    angle_losses = torch.minimum(torch.abs(outputs[:, 2]-labels[:, 2]), 2*np.pi-torch.abs(outputs[:, 2]-labels[:, 2]))**2
    return torch.mean(position_losses+angle_losses)


if __name__ == '__main__':
    outputs_test = torch.tensor([[76.68195, 8.009406, -2.5358713]])
    labels_test = torch.tensor([[76.68195, 8.009405, -2.5365684]])
    print(custom_loss(outputs_test, labels_test))
