import torch
from model import Net, transform, trainset
from PIL import Image

net = Net()
state_dict = torch.load('./friday_net.pth', map_location='cpu')
net.load_state_dict(state_dict)
net.eval()

im = transform(Image.open('test.jpg')).reshape((-1, 3, 224, 224))

out = net(im)

probs = out/out.sum()

for klass, prob in zip(trainset.classes, probs[0]):
    print(klass, prob)
