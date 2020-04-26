import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
import os

from model import Net, trainset, valset

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=multiprocessing.cpu_count())
valloader = torch.utils.data.DataLoader(
    valset, batch_size=32, shuffle=True, num_workers=multiprocessing.cpu_count())

SAVE_EVERY = 10

net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

print("training...")

def train(train, loader):
    num_examples = 0
    total_acc = 0.0
    total_loss = 0.0

    net.train(train)

    for x, y in loader:
        x = x.cuda()
        y = y.cuda()

        if train:
            optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        acc = (out.argmax(1) == y).float().sum() / len(y)

        total_acc += acc * len(y)
        total_loss += loss
        num_examples += len(y)

    return total_acc/num_examples, total_loss/num_examples, num_examples


if __name__ == "__main__":
    for epoch in range(100):
        print(f"epoch {epoch}")
        acc, loss, num_examples = train(train=True, loader=trainloader)
        print(f"  - train: acc {acc}, loss {loss}, num_examples {num_examples}")

        with torch.no_grad():
            acc, loss, num_examples = train(train=False, loader=valloader)
        print(f"  - val: acc {acc}, loss {loss}, num_examples {num_examples}")

        if (epoch % SAVE_EVERY) == (SAVE_EVERY-1):
            PATH = './friday_net.pth'
            print(f"saving to {PATH}")
            torch.save(net.state_dict(), PATH)
