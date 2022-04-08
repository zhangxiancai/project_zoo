'''
pt转onnx
'''

import os

import torch
from model import SixDRepNet,Resnet18_face_angle
from torch.autograd import Variable
from torchvision import models

pt_path='/home/xiancai/face_angle/6DRepNet/results/2022_04_07/RepVGG-A0s_epoch_180_mae8.0871_transfer.pth'
onnx_path=pt_path[:-4]+'.onnx'

# pt_model_path='/home/xiancai/face_angle/6DRepNet/results/other/resnet18-f37072fd.pth'
# onnx_path='/home/xiancai/face_angle/6DRepNet/results/other/A0-Half-debug.onnx'

device=torch.device('cuda')
# model
model = SixDRepNet(backbone_name='RepVGG-A0s',
                   backbone_file='',
                   deploy=True,
                   pretrained=False,
                   export_onnx=True)
# model = Resnet18_face_angle(export_onnx=True)
# model = models.resnet18()

# weights
saved_state_dict = torch.load(pt_path)

if 'model_state_dict' in saved_state_dict:
	model.load_state_dict(saved_state_dict['model_state_dict'])
else:
	model.load_state_dict(saved_state_dict)

model.eval()
model.to(device)

# to onnx
dummy_input = Variable(torch.randn(1, 3, 112, 112)).to(device)
# dummy_input = Variable(torch.randn(1, 3, 84, 84)).to(device)
torch.onnx.export(model, dummy_input, onnx_path, verbose = True, opset_version=11, input_names = ['img'], output_names = ['6d'])

os.system(f'python3 -m onnxsim {onnx_path} {onnx_path[:-5]}_sim.onnx')

# macs, params
from thop import profile, clever_format
macs, params = profile(model, inputs=(dummy_input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f'macs:{macs},params:{params}')