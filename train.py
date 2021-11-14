import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
import os

from model import QATNet, Net, trainset, valset

workers = 16
epochs = 40


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True
)

print("creating net")

net = Net()
print(net)
net.cuda()
pad_weights, friday_weights = trainset.weights()
pad_criterion = nn.CrossEntropyLoss(
    weight=pad_weights.cuda(),
)
friday_criterion = nn.CrossEntropyLoss(
    weight=friday_weights.cuda(),
)
friday_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.0001)

# schedulers

# achieved 90+
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(trainloader), epochs=epochs)

# acc 0.9103773832321167, lr .0005, epoch 13
# acc 0.9811 epoch 39
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.0005, steps_per_epoch=len(trainloader), epochs=epochs
)

# 0.85 after 19 epochs
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00025, steps_per_epoch=len(trainloader), epochs=epochs)

# 0.896 after 5 epochs
# epochs = 20
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(trainloader), epochs=epochs)

# 0.886 after 14 epochs
# epochs = 20
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, steps_per_epoch=len(trainloader), epochs=epochs)


print("training...")


def train(train, loader):
    num_examples = 0
    total_pad_acc = torch.tensor(0.0).cuda()
    total_friday_acc = torch.tensor(0.0).cuda()
    total_loss = torch.tensor(0.0).cuda()

    net.train(train)

    for img, pad, friday in loader:
        img = img.cuda()
        pad = pad.cuda()
        friday = friday.cuda()

        if train:
            optimizer.zero_grad()
        pad_out = net(img)
        #pad_out, friday_out = net(img)
        loss = pad_criterion(pad_out, pad)
        #loss += friday_criterion(friday_out, friday)
        if train:
            loss.backward()
            optimizer.step()
        pad_acc = (pad_out.argmax(1) == pad).float().sum() / len(pad)
        #friday_acc = (friday_out.argmax(1) == friday).float().sum() / len(pad)

        total_pad_acc += pad_acc * len(pad)
        #total_friday_acc += friday_acc * len(pad)
        total_loss += loss
        num_examples += len(pad)

        if train:
            scheduler.step()

    return (
        total_pad_acc / num_examples,
        total_friday_acc / num_examples,
        total_loss / num_examples,
        num_examples,
    )


def get_lr(optimizer):
    return torch.mean(torch.tensor([group["lr"] for group in optimizer.param_groups]))


def main():
    best_acc = 0
    for epoch in range(epochs):
        print(f"epoch {epoch} - lr {get_lr(optimizer):.10f}")
        pad_acc, friday_acc, loss, num_examples = train(train=True, loader=trainloader)
        print(
            f"  - train: pad_acc {pad_acc}, friday_acc {friday_acc}, loss {loss}, num_examples {num_examples}"
        )

        with torch.no_grad():
            pad_acc, friday_acc, loss, num_examples = train(
                train=False, loader=valloader
            )
        print(
            f"  - val: pad_acc {pad_acc}, friday_acc {friday_acc}, num_examples {num_examples}"
        )

        if pad_acc > best_acc:
            best_acc = pad_acc
            PATH = "./friday_net.pth"
            print(f"saving to {PATH}, acc {pad_acc}")
            torch.save(net.state_dict(), PATH)

        # scheduler.step(loss)


if __name__ == "__main__":
    main()
