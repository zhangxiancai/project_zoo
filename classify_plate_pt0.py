'''
#!/usr/bin/env python3
测试车牌识别模型 pt0 /home/xiancai/LPRNet_Pytorch/model/LPRNet0.py
'''
import os
import re
import shutil
import time

import onnxruntime as ort
import cv2
import numpy as np

def remove_repeate_blank_label(pre):
    '''
    CTC:删除重复和占位符
    :param pre:  1*18 np
    :return:
    '''

    pre=pre[0].tolist()
    no_repeat_blank_label = list()
    pre.insert(0,len(CHARS0)-1)
    p = pre[0]
    if p != len(CHARS0) - 1:
        no_repeat_blank_label.append(p)
    for c in pre:  # dropout repeate label and blank label 注意
        if (c == p) or (c == len(CHARS0) - 1): # 如果c与前一个字符相同或是占位符
            pass
        else:
            no_repeat_blank_label.append(c)
        p = c

    # # debug
    # no_repeat_blank_label_pt=list()
    # pre_c = pre[0]
    # for c in pre:  # dropout repeate label and blank label 注意
    #     if (pre_c == c) or (c == len(CHARS) - 1):
    #         if c == len(CHARS) - 1:
    #             pre_c = c
    #         continue
    #     no_repeat_blank_label_pt.append(c)
    #     pre_c = c
    # if no_repeat_blank_label!=no_repeat_blank_label_pt:
    #     print(pre)
    #     print(no_repeat_blank_label)
    #     print(no_repeat_blank_label_pt)
    #     raise

    return no_repeat_blank_label


def classify_plate(img_path):
    '''
    车牌识别接口 pt
    :param img_path:
    :return: 1*~ list int
    '''
    img=cv2.imread(img_path) # to h,w,c
    img=cv2.resize(img,(94,24)) # w=94,h=24
    # img=img[...,::-1] # to RGB
    img=img.transpose(2, 0, 1) # to 3*h*w
    img=img[None,...] # to 1*3*w*h
    img = np.ascontiguousarray(img).astype(np.float32) # to float
    img=(img-127.5)/128 # to [-1,1]

    # pre=ses.run(None,{'data':img})[0]
    # # print(pre.shape)
    pre=model_pt(torch.tensor(img)).detach().numpy() # 已mean
    if img_path.split('/')[-1] == '0066-1_1-302&495_428&539-428&536_304&539_302&498_426&495-0_0_33_6_26_24_30-69-2-皖A9G206.jpg':
        print(pre)

    pre=np.argmax(pre,axis=1)
    # res_lab = CHARS[pre]
    res=remove_repeate_blank_label(pre) #删除重复和占位符 pre：['苏' 'E' '-' '-' '-' '6' '-' '-' '3' '-' '-' '7' '-' '-' 'A' '-' '-' 'F']
    # res_lab=CHARS[res]
    # print(res_lab)
    return res


from model.LPRNet0 import CHARS0
CHARS0=np.array(CHARS0)
CHARS_DICT = {char:i for i, char in enumerate(CHARS0)}

model_path='/home/xiancai/LPRNet_Pytorch/result/origin/Final_LPRNet_model.pth'

# pt model init
from model.LPRNet0 import build_lprnet
model_pt = build_lprnet(class_num=len(CHARS0),dropout_rate=0).cpu()
# load weights
import torch
from collections import OrderedDict
checkpoint = torch.load(model_path)
state_dict_rename = OrderedDict()
for k, v in checkpoint.items():
	if k.startswith('module.'):
		name = k[7:]
	else:
		name = k
	state_dict_rename[name] = v
model_pt.load_state_dict(state_dict_rename, strict=False)
model_pt.eval()

# ss=classify_plate('/data1/xiancai/PLATE_DATA/res_classify_err/2022_01_12_best_adjust_0.9811.onnx-PLATE_DATA_plate_classify_dataset_adjust_xxx_0.9485/301684136284722222-86_90-288&453_475&543-475&519_288&543_288&467_468&453-0_0_5_24_31_27_33_33-72-73-皖AF07399-pre皖AF0739.jpg')

if __name__=='__main__':
    ''' 
    测试车牌识别模型pt，计算精度
    测试一个文件夹的图片，图片名称格式 ~-川JK0707.jpg  或val.txt
    错误图片保存至err_path
    '''
    # load data
    imgs_path = '/data1/xiancai/PLATE_DATA/plate_classify_dataset/val.txt'
    DIR_TAG=False
    if os.path.isdir(imgs_path):
        ls=os.listdir(imgs_path)
        DIR_TAG=True
    else:
        with open(imgs_path,'r') as f:
            ls=list(map(lambda x:x.strip(),f.readlines()))
            ls=ls[:1000]

    # test
    errs=[]
    tag=0
    for ind,i in enumerate(ls):
        if not DIR_TAG:
            ii=i.split('/')[-1]
            imgs_path=i[:len(i)-len(ii)]
            i=ii
            # print(imgs_path)
            # print(i)
        t0=time.time()
        pre=classify_plate(imgs_path+i) # classify
        t1=time.time()

        # count
        pre_lab=''.join(CHARS0[pre])
        tru_lab = re.split('-|_', i)[-1][:-4].strip()  #????????deepcamdata clean
        if pre_lab!=tru_lab:
            errs.append((imgs_path,i,pre_lab))
            tag+=1
        print(f'{ind}/{len(ls)},inference:{t1-t0}s')
    acc=1-tag/len(ls)
    print(f'acc:{acc}  {tag}/{len(ls)}')

    # save
    sub_dir = '_'.join(model_path.split('/')[-2:]) + '--' + '_'.join(imgs_path.split('/')[-3:-1])+'--'+f'acc{round(acc,4)}'
    err_path=f'/data1/xiancai/PLATE_DATA/res_classify_err/{sub_dir}/'
    if not os.path.exists(err_path):
        os.mkdir(err_path)
    for imgs_path,i,pre_lab in errs:
        shutil.copy(imgs_path + i, err_path + i[:-4] + '-' + 'pre' + pre_lab + '.jpg')
        print(f'err img saved to {err_path}')