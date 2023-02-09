import numpy as np
import torch

from model import Net, xyrad2xysincos
import prepare_dataset
from dataset_iterator import dataset_iterator


epoch_len = 100
batch_size = 40
iterations = 200


grid_data = prepare_dataset.make_grid_data()
train_data, train_labels = prepare_dataset.get_data_and_labels(grid_data)


train_data = torch.from_numpy(train_data)
train_data = xyrad2xysincos(train_data)
train_labels = torch.from_numpy(train_labels)
train_labels = xyrad2xysincos(train_labels)


net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(iterations):
    running_loss = 0.
    data_loader = dataset_iterator(input_data=train_data, label_data=train_labels, batch_size=batch_size)
    for i in range(epoch_len):
        inputs, labels = next(data_loader)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / epoch_len:.3f}')

torch.save(net.state_dict(), "trained.pth")
