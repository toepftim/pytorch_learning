import torch

from model import Net, xydeg2xysincos
import prepare_dataset


dataset_size = 50000
batch_size = 100
iterations = 10


train_data = prepare_dataset.make_random_data(dataset_size)
train_labels = prepare_dataset.get_labels(train_data)

train_data = torch.from_numpy(train_data)
train_data = xydeg2xysincos(train_data)
train_labels = torch.from_numpy(train_labels)
train_labels = xydeg2xysincos(train_labels)

net = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(iterations):
    running_loss = 0.
    for i in range(dataset_size//batch_size):
        inputs = train_data[i*batch_size:(i+1)*batch_size]
        labels = train_labels[i*batch_size:(i+1)*batch_size]

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

torch.save(net.state_dict(), "trained.pth")
