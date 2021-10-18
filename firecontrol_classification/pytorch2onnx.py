import torch
from torch.autograd import Variable
from backbones.resnet import resnet18
from collections import OrderedDict

output_onnx = 'r18.onnx'
pytorch_model = 'checkpoints/resnet18/resnet18-13-best.pth'
'''
Epoch:13 Iter:0/68 LR:0.0398 Loss:0.0006	
Epoch:13 Iter:50/68 LR:0.0397 Loss:0.0007	
Test set: loss: 0.0511, Accuracy: 0.9136
'''

model = resnet18().cpu()
checkpoint = torch.load(pytorch_model, map_location={'cuda:6':'cuda:0'})
state_dict_rename = OrderedDict()
for k, v in checkpoint.items():
	if k.startswith('module.'):
		name = k[7:]
	else:
		name = k
	state_dict_rename[name] = v
model.load_state_dict(state_dict_rename, strict=False)
model.train(False)
dummy_input = Variable(torch.randn(1, 3, 112, 112)).cpu()
torch.onnx.export(model, dummy_input, output_onnx, verbose = True, opset_version=11, input_names = ['data'], output_names = ['fc5'])

