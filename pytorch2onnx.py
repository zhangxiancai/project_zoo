import torch
from torch.autograd import Variable
from collections import OrderedDict
from model.LPRNet import LPRNet, CHARS

# from model.LPRNet0 import build_lprnet,CHARS0
import onnx
import onnxruntime as ort

output_onnx = '/home/xiancai/LPRNet_Pytorch/result/2022_02_21/best_0.9816.onnx'
pytorch_model = '/home/xiancai/LPRNet_Pytorch/result/2022_02_21/best_0.9816.pth'


model =  LPRNet(class_num=len(CHARS), dropout_rate=0, export=True).cpu()
# model = build_lprnet(class_num=len(CHARS0),dropout_rate=0).cpu()

checkpoint = torch.load(pytorch_model, map_location={'cuda:6':'cuda:0'})
state_dict_rename = OrderedDict()
for k, v in checkpoint.items():
	if k.startswith('module.'):
		name = k[7:]
	else:
		name = k
	state_dict_rename[name] = v
model.load_state_dict(state_dict_rename, strict=False)
model.eval()
dummy_input = Variable(torch.randn(1, 3, 24, 94)).cpu()
torch.onnx.export(model, dummy_input, output_onnx, verbose = True, opset_version=11, input_names = ['data'], output_names = ['container'])

# check

ses=ort.InferenceSession(output_onnx)
pre_pt=model(dummy_input).detach().numpy()
inp=dummy_input.numpy()
pre_onnx=ses.run(None,{'data':inp})[0] #1*70*4*18
print(abs(pre_onnx-pre_pt).max())


