import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 80)
        self.fc2 = torch.nn.Linear(80, 50)
        self.fc3 = torch.nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.cat((x[:, :2], F.hardtanh(x[:, 2:4])), 1)
        return x


def xyrad2xysincos(deg_input: torch.Tensor):
    angles = deg_input[:, 2:3]
    sins = torch.sin(angles)
    coses = torch.cos(angles)
    return torch.cat((deg_input[:, :2], sins, coses), 1)


def xysincos2xyrad(sincos_output: torch.Tensor):
    out_angles = torch.atan2(sincos_output[:, 2:3], sincos_output[:, 3:4])
    return torch.cat((sincos_output[:, :2], out_angles), 1)
