import torch
import torchvision
from model import QATNet, Net, valset
from PIL import Image
import os
from tqdm import tqdm
from contextlib import contextmanager
import time

@contextmanager
def measure(name):
    print(f"measuring {name}")
    start = time.time()
    yield
    print(f"{name} took {time.time() - start}")

dataloader = torch.utils.data.DataLoader(
    valset, batch_size=16, shuffle=True, num_workers=16
)

def dequantize_state_dict(state_dict):
    new_dict = {}
    drop_keys = {
        "weight_fake_quant",
        "qconfig",
        "activation_post_process",
    }
    for key, value in state_dict.items():
        keep = True
        for drop in drop_keys:
            if drop in key:
                keep = False
                break
        if keep:
            new_dict[key] = value
    return new_dict


state_dict = torch.load("./friday_net.pth", map_location="cpu")

# qat_net = QATNet()
# qat_net.load_state_dict(state_dict)

net = Net()
net.load_state_dict(dequantize_state_dict(state_dict))
net.eval()

net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
net.fuse_model()
model_fp32_prepared = torch.quantization.prepare(net)

with measure("fp32 prepare"):
    for img, pad, friday in tqdm(dataloader):
        model_fp32_prepared(img)

model_int8 = torch.quantization.convert(model_fp32_prepared)
print(model_int8)
print(model_int8(img))
torch.jit.save(torch.jit.script(model_int8), 'friday_net_quant_jit.pt')
jitted = torch.jit.load('friday_net_quant_jit.pt')

with torch.no_grad():
    with measure("int8 infer"):
        for img, pad, friday in tqdm(dataloader):
            model_int8(img)

    with measure("int8 jit infer"):
        for img, pad, friday in tqdm(dataloader):
            jitted(img)

    with measure("int8 jit infer2"):
        for img, pad, friday in tqdm(dataloader):
            jitted(img)
