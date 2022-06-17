'''调试pt模型和onnx模型'''

from train_LPRNet import Greedy_Decode_Eval,Greedy_Decode_Eval_debug
from classify_plate import classify_plate

from model.LPRNet import LPRNet, CHARS
from collections import OrderedDict
import torch
import argparse
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader

onnx_model = '/home/xiancai/LPRNet_Pytorch/result/2022_01_12/best_adjust_0.9811.onnx'
# onnx_model = '/home/xiancai/LPRNet_Pytorch/LPRNet.onnx'
pytorch_model = '/home/xiancai/LPRNet_Pytorch/result/2022_01_12/best_adjust_0.9811.pth'

test_img_dirs = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'

model_pt =  LPRNet(class_num=len(CHARS), dropout_rate=0, export=False).cpu()

checkpoint = torch.load(pytorch_model)
state_dict_rename = OrderedDict()
for k, v in checkpoint.items():
	if k.startswith('module.'):
		name = k[7:]
	else:
		name = k
	state_dict_rename[name] = v
model_pt.load_state_dict(state_dict_rename, strict=False)
model_pt.eval()

parser = argparse.ArgumentParser(description='parameters to train net')
parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
args = parser.parse_args()
test_dataset = LPRDataLoader(test_img_dirs, [94, 24], 8, augment=False) # collate_fn(batch)
# test_acc=Greedy_Decode_Eval(model_pt, test_dataset, args) # test pt model

tag=Greedy_Decode_Eval_debug(model_pt, test_dataset, args) # test pt model

print(f'pt_acc:{tag}')


# import classify_plate
# with open(test_img_dirs, 'r') as f:
# 	ls = list(map(lambda x: x.strip(), f.readlines()))
# debug_res_onnx=[]
# for i in ls:
# 	res=classify_plate.classify_plate(i)
# 	debug_res_onnx.append(res)
#
# # check
# for i in range(len(debug_res_pt)):
# 	if debug_res_pt[i]!=debug_res_onnx[i]:
# 		print(debug_res_pt[i],debug_res_onnx[i])
#
# print()


