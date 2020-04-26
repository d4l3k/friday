import torch
from converters import pytorch2savedmodel, savedmodel2tflite
from model import Net

net = Net()
state_dict = torch.load('./friday_net.pth', map_location='cpu')
net.load_state_dict(state_dict)
net.eval()

dummy_input = torch.randn(1, 3, 224, 224)
input_names = ['image_array']
output_names = ['category']

print('onnx export')
onnx_model_path = 'model.onnx'
torch.onnx.export(net, dummy_input, onnx_model_path,
		  input_names=input_names, output_names=output_names)

print('keras export')
saved_model_dir = 'saved_model'
pytorch2savedmodel(onnx_model_path, saved_model_dir)

print('tflite export')
tflite_model_path = 'model.tflite'
tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)

print('tflite quantized export')
tflite_quantized_model_path = 'model_quantized.tflite'
tflite_quantized_model = savedmodel2tflite(saved_model_dir, tflite_quantized_model_path, quantize=True)
