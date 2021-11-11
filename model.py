import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import models
from torchvision import transforms
from enum import Enum
import xxhash
import glob
from os import path
from datetime import datetime
from PIL import Image
from collections import defaultdict


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_transform = transforms.Compose(
    [
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transform,
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transform,
    ]
)


def is_valid(filename):
    return ".jpg" in filename or ".png" in filename


def is_validation(filename):
    name = filename.split(".")[1]
    h = xxhash.xxh64(name).intdigest()
    return is_valid(filename) and h % 10 == 0


def is_train(filename):
    return is_valid(filename) and not is_validation(filename)


class FridayLabel(Enum):
    UNKNOWN = 0
    PREPOOP = 1


class PadLabel(Enum):
    CLEAR = 0
    NEWPEE = 1
    OLDPEE = 2
    POOP = 3


PAD_LABELS = [label.name.lower() for label in PadLabel]


def get_file_class(filename) -> int:
    parts = filename.split("/")
    idx = PAD_LABELS.index(parts[-2])
    assert idx >= 0, filename
    return idx


def get_file_time(filename) -> float:
    base = path.basename(filename).split(".")[0]
    try:
        return int(datetime.fromisoformat(base).timestamp())
    except ValueError:
        return None


class MultiTaskDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, root: str, transform, is_valid_file):
        self.transform = transform
        files = glob.glob(path.join(root, "*/*"))
        poop_times = set()
        for f in files:
            pad = get_file_class(f)
            if pad == PadLabel.POOP.value:
                poop_times.add(get_file_time(f))

        self.examples = []

        for f in files:
            if not is_valid_file(f):
                continue

            pad = get_file_class(f)
            friday = FridayLabel.UNKNOWN.value
            if pad != PadLabel.POOP.value:
                ftime = get_file_time(f)
                if ftime and ((ftime + 1) in poop_times or (ftime + 2) in poop_times):
                    friday = FridayLabel.PREPOOP.value

            self.examples.append((f, pad, friday))

        pad_counts = defaultdict(lambda: 0)
        friday_counts = defaultdict(lambda: 0)
        for _, pad, friday in self.examples:
            pad_counts[pad] += 1
            friday_counts[friday] += 1

        print(f"friday {friday_counts}, pad {pad_counts}")

    def weights(self):
        pad_counts = torch.zeros(len(PadLabel))
        friday_counts = torch.zeros(len(FridayLabel))
        for _, pad, friday in self.examples:
            pad_counts[pad] += 1
            friday_counts[friday] += friday

        print(f"pad counts {pad_counts}")
        print(f"friday counts {friday_counts}")

        pad_weights = 1/(pad_counts/pad_counts.mean()) * torch.tensor([1, 1, 1, 2])
        friday_weights = 1/(friday_counts/friday_counts.mean())

        print(f"pad weights {pad_weights}")
        print(f"friday weights {friday_weights}")


        return pad_weights, friday_weights


    def __getitem__(self, index):
        filename, pad, friday = self.examples[index]
        img = Image.open(filename)
        return self.transform(img), pad, friday

    def __len__(self):
        return len(self.examples)


trainset = MultiTaskDataset(
    root="./data", transform=train_transform, is_valid_file=is_train
)
valset = MultiTaskDataset(
    root="./data", transform=val_transform, is_valid_file=is_validation
)

NUM_CLASSES = len(PAD_LABELS)
NUM_FRIDAY_CLASSES = len(FridayLabel)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(0.2),
        )
        self.pad = nn.Linear(self.cnn.last_channel, NUM_CLASSES)
        #self.friday = nn.Linear(self.cnn.last_channel, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        return self.pad(x)
        #return self.pad(x), self.friday(x)


def QATNet():
    net = Net()
    # quantize aware training
    net.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(net, inplace=True)
    return net
