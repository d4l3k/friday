from captum.attr import IntegratedGradients
import sys

import numpy as np
from model import Net, val_transform, trainset, transform
import torch
from PIL import Image
import torchvision.transforms.functional as tf

torch.manual_seed(123)
np.random.seed(123)

path = sys.argv[-1]

net = Net()
state_dict = torch.load("./friday_net.pth", map_location="cpu")
net.load_state_dict(state_dict)
net.eval()


def save_img(im, path):
    print(f"saving {path}...")
    max, min = torch.max(im), torch.min(im)
    im = (im - min) / (max - min)
    out_im = tf.to_pil_image((im.squeeze(0) * 255).byte())
    out_im.save(path)


im = val_transform(Image.open(path)).unsqueeze_(0)
print(im.size())
save_img(im, "ig.png")
baseline = torch.zeros(im.size())

out = net(im)
klass = torch.argmax(out).item()
print("predicted", trainset.classes[klass], out)

for i, name in enumerate(trainset.classes):
    ig = IntegratedGradients(net)
    attributions, delta = ig.attribute(
        im, baseline, target=i, return_convergence_delta=True
    )

    # print('IG Attributions:', attributions)
    print("Convergence Delta:", delta)

    save_img(attributions, f"ig_{name}.png")
    if i == klass:
        save_img(attributions, "ig_attributions.png")
