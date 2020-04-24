import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F

transform = transforms.Compose( [
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.ImageFolder(root='./data', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8)

num_classes = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Sequential(
          nn.Dropout(0.2),
          nn.Linear(self.cnn.last_channel, num_classes)
        )

    def forward(self, x):
        return self.cnn(x)

net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for x, y in trainloader:
    x = x.cuda()
    y = y.cuda()

    optimizer.zero_grad()
    out = net(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    acc = (out.argmax(1) == y).float().sum() / len(y)
    print(acc.item(), loss.item(), y)



