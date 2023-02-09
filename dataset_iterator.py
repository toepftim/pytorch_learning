import torch
import random


def dataset_iterator(input_data: torch.Tensor, label_data: torch.Tensor, batch_size: int):
    dataset_size = input_data.size()[0]
    options = range(dataset_size)
    weights = [10]*dataset_size
    chosen = 0
    while True:
        idxs = random.choices(options, weights=weights, k=batch_size)
        for i in idxs:
            weights[i] -= 1
        chosen += batch_size
        if chosen >= dataset_size:
            for i in range(len(weights)):
                weights[i] += chosen//dataset_size
            chosen %= dataset_size
        inputs = input_data[idxs]
        labels = label_data[idxs]
        random_bonus = torch.atan2(labels[:, 2:3], labels[:, 3:4])
        random_bonus = 2*(random_bonus/torch.pi + 1)
        random_bonus = random_bonus + torch.rand(random_bonus.size())
        random_bonus %= 4
        real_random_bonus = torch.rand(random_bonus.size())*4
        random_bonus = torch.where(1 > inputs[:, 0:1], real_random_bonus, random_bonus)
        random_bonus = torch.where(6 < inputs[:, 0:1], real_random_bonus, random_bonus)
        random_bonus = torch.where(1 > inputs[:, 1:2], real_random_bonus, random_bonus)
        random_bonus = torch.where(6 < inputs[:, 1:2], real_random_bonus, random_bonus)
        inputs = torch.cat((inputs, random_bonus), 1)
        yield inputs, labels


