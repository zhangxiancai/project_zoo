import torch
from torch.autograd import Variable
from models.resnet import *


output_onnx = 'maskface.onnx'
pytorch_model = 'checkpoints/model-epoch-5.pth'

model = resnet50()
model.cuda()

#checkpoint = torch.load(pytorch_model, map_location={'cuda:6':'cuda:0'})
#model.load_state_dict(torch.load(pytorch_model))
model.train(False)
dummy_input = Variable(torch.randn(1, 3, 112, 112)).cuda()
torch.onnx.export(model, dummy_input, output_onnx, verbose = True, input_names = ['data'], output_names = ['fc2'])

