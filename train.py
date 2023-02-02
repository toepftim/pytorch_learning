import torch

from model import Net
import prepare_dataset


dataset_size = 10000
batch_size = 1
iterations = 2


train_data = prepare_dataset.make_random_data(dataset_size)
train_labels = prepare_dataset.get_labels(train_data)

net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(iterations):
    running_loss = 0.
    for i in range(dataset_size//batch_size):
        inputs = torch.from_numpy(train_data[i*batch_size:(i+1)*batch_size])
        labels = torch.from_numpy(train_labels[i*batch_size:(i+1)*batch_size])

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

torch.save(net.state_dict(), "trained.pth")
