from torch import nn
import torchvision
from torchvision import models
from torchvision import transforms



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    transform,
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transform,
])

def is_valid(filename):
    return '.jpg' in filename or '.png' in filename

def is_validation(filename):
    name = filename.split('.')[1]
    return is_valid(filename) and hash(name) % 10 == 0

def is_train(filename):
    return is_valid(filename) and not is_validation(filename)

trainset = torchvision.datasets.ImageFolder(
    root='./data', transform=train_transform, is_valid_file=is_train)
valset = torchvision.datasets.ImageFolder(
    root='./data', transform=val_transform, is_valid_file=is_validation)

NUM_CLASSES = len(trainset.classes)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.cnn.last_channel, NUM_CLASSES),
        )

    def forward(self, x):
        return self.cnn(x)


