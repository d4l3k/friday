import torch
from converters import pytorch2savedmodel, savedmodel2tflite
import torchvision
from model import QATNet, Net, val_transform
from tflite import get_tflite_outputs
from PIL import Image
import os



dataset = torchvision.datasets.ImageFolder(root='./data', transform=val_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

def to_tflite_input(tensor):
    return tensor.numpy().transpose((0, 2, 3, 1))

def representative_dataset_gen():
    for i, (x, _) in enumerate(dataloader):
        yield [to_tflite_input(x)]
        if i > 200:
            return

def dequantize_state_dict(state_dict):
    new_dict = {}
    drop_keys = {
        'weight_fake_quant',
        'qconfig',
        'activation_post_process',
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

state_dict = torch.load('./friday_net.pth', map_location='cpu')

#qat_net = QATNet()
#qat_net.load_state_dict(state_dict)

net = Net()
net.load_state_dict(dequantize_state_dict(state_dict))
net.eval()
#net=torch.quantization.convert(qat_net.eval(), inplace=False)

im = Image.open('./test.jpg')
data = val_transform(im).reshape((1, 3, 224, 224))
tflite_input = to_tflite_input(data)
print(tflite_input, tflite_input.mean(), tflite_input.std())
torch_output = net(data)
#torch_quantize_output = qat_net(data)

dummy_input = torch.randn(1, 3, 224, 224)
input_names = ['image_array']
output_names = ['category']

print('onnx export')
onnx_model_path = 'model.onnx'
torch.onnx.export(net, dummy_input, onnx_model_path, input_names=input_names, output_names=output_names)

print('keras export')
saved_model_dir = 'saved_model'
pytorch2savedmodel(onnx_model_path, saved_model_dir)

print('tflite export')
tflite_model_path = 'model.tflite'
tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)


print('tflite quantized export')
tflite_quantized_model_path = 'model_quantized.tflite'
tflite_quantized_model = savedmodel2tflite(saved_model_dir,
        tflite_quantized_model_path, quantize=True,
        representative_dataset=representative_dataset_gen)

print('edgetpu_compiler')
os.system(f"edgetpu_compiler {tflite_quantized_model_path}")

tflite_output = get_tflite_outputs(tflite_input, tflite_model).reshape(-1, )
tflite_quantized_output = get_tflite_outputs(tflite_input, tflite_quantized_model).reshape(-1, )
print('torch', torch_output)
#print('torch_quantized', torch_output)
print('tflite', tflite_output)
print('tflite_quantized', tflite_quantized_output)
