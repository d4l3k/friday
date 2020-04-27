import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision import transforms
import xxhash



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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
    h = xxhash.xxh64(name).intdigest()
    return is_valid(filename) and h % 10 == 0

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


def QATNet():
    net = Net()
    # quantize aware training
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(net, inplace=True)
    return net
