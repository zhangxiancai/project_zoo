import torch
from torch.autograd import Variable
from backbones.resnet import resnet18
from collections import OrderedDict

output_onnx = '/home/xiancai/fire-equipment-demo/firecontrol_classification/Result/2021_12_06/fire_classify_resnet18_cls114_12_06.onnx'
pytorch_model = '/home/xiancai/fire-equipment-demo/firecontrol_classification/Result/2021_12_06/fire_classify_resnet18_cls114_12_06.pth'
'''
Epoch:13 Iter:0/68 LR:0.0398 Loss:0.0006	
Epoch:13 Iter:50/68 LR:0.0397 Loss:0.0007	
Test set: loss: 0.0511, Accuracy: 0.9136
'''

model = resnet18(num_classes=114).cpu()
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

# macs, params
from thop import profile, clever_format
macs, params = profile(model, inputs=(dummy_input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f'macs:{macs},params:{params}')