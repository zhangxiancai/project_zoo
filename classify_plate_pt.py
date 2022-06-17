'''
#!/usr/bin/env python3
测试车牌识别模型 pt
'''
import glob
import os
import re
import shutil
import time

import onnxruntime as ort
import cv2
import numpy as np
import torch


class inference:

    model_path = '/home/xiancai/plate/LPRNet_Pytorch/Result/2022_05_23/best_0.9826.pth'
    # model_path = '/home/xiancai/plate/LPRNet_Pytorch/Result/2022_04_20_more_agu_data/best_0.9828.pth'
    # model_path = '/home/xiancai/plate/LPRNet_Pytorch/weights/1650613218/best_0.9946.pth'
    # model_path = '/home/xiancai/plate/LPRNet_Pytorch/weights/1650781812/best_0.9976.pth'
    # model_path = '/home/xiancai/plate/LPRNet_Pytorch/Result/2022_02_21/best_0.9816.pth'
    CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新', '学', '警',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'I', 'O', '-'
             ]
    CHARS = np.array(CHARS)
    CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
    # pt model init
    from model.LPRNet import LPRNet,LSTM_LPRNet
    # model_pt = LSTM_LPRNet(class_num=len(CHARS), dropout_rate=0, export=False).cpu()
    model_pt = LPRNet(class_num=len(CHARS), dropout_rate=0, export=False).cpu()
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


    def infer(self, img_path):
        '''
        车牌识别接口 pt
        :param img_path:
        :return: 1*~ list int
        '''
        img = cv2.imread(img_path)  # to h,w,c
        img = cv2.resize(img, (94, 24))  # w=94,h=24
        # img=img[...,::-1] # to RGB
        img = img.transpose(2, 0, 1)  # to 3*h*w
        img = img[None, ...]  # to 1*3*w*h
        img = np.ascontiguousarray(img).astype(np.float32)  # to float
        img = (img - 127.5) / 128  # to [-1,1]

        # pre=ses.run(None,{'data':img})[0]
        # # print(pre.shape)
        pre = self.model_pt(torch.tensor(img)).detach().numpy()  # 已mean
        pre = np.argmax(pre, axis=1)
        # res_lab = CHARS[pre]
        res = self.remove_repeate_blank_label(pre)  # 删除重复和占位符 pre：['苏' 'E' '-' '-' '-' '6' '-' '-' '3' '-' '-' '7' '-' '-' 'A' '-' '-' 'F']
        # res_lab=CHARS[res]
        # print(res_lab)
        return res

    def remove_repeate_blank_label(self,pre):
        '''
        CTC:删除重复和占位符
        :param pre:  1*18 np
        :return:
        '''
        pre=pre[0].tolist()
        no_repeat_blank_label = list()
        pre.insert(0,69)
        p = pre[0]
        if p != len(self.CHARS) - 1:
            no_repeat_blank_label.append(p)
        for c in pre:  # dropout repeate label and blank label 注意
            if (c == p) or (c == len(self.CHARS) - 1): # 如果c与前一个字符相同或是占位符
                pass
            else:
                no_repeat_blank_label.append(c)
            p = c

        return no_repeat_blank_label

class test:

    def temp(self):
        '''
            测试车牌识别模型pt，计算精度
            测试一个文件夹的图片，图片名称格式 ~-川JK0707.jpg  或val.txt
            错误图片保存至err_path
            '''

        inf = inference()
        # load data
        imgs_path = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/train.txt'
        DIR_TAG = False
        if os.path.isdir(imgs_path):
            ls = os.listdir(imgs_path)
            DIR_TAG = True
        else:
            with open(imgs_path, 'r') as f:
                ls = list(map(lambda x: x.strip(), f.readlines()))
                # ls = ls[:1000]

        # test
        errs = []
        tag = 0
        for ind, i in enumerate(ls):
            if not DIR_TAG:
                ii = i.split('/')[-1]
                imgs_path = i[:len(i) - len(ii)]
                i = ii
                # print(imgs_path)
                # print(i)
            t0 = time.time()
            pre = inf.infer(imgs_path + i)  # classify
            t1 = time.time()

            # count
            pre_lab = ''.join(inf.CHARS[pre])
            tru_lab = re.split('-|_', i)[-1][:-4].strip()  # ????????deepcamdata clean
            if pre_lab != tru_lab:
                errs.append((imgs_path, i, pre_lab))
                tag += 1
            print(f'{ind}/{len(ls)},inference:{t1 - t0}s')
        acc = 1 - tag / len(ls)
        print(f'acc:{acc}  {tag}/{len(ls)}')

        # save
        sub_dir = '_'.join(inf.model_path.split('/')[-2:]) + '--' + '_'.join(
            imgs_path.split('/')[-3:-1]) + '--' + f'acc{round(acc, 4)}'
        err_path = f'/data1/xiancai/PLATE_DATA/res_classify_err/{sub_dir}/'
        if not os.path.exists(err_path):
            os.mkdir(err_path)
        for imgs_path, i, pre_lab in errs:
            shutil.copy(imgs_path + i, err_path + i[:-4] + '-' + 'pre' + pre_lab + '.jpg')
            print(f'err img saved to {err_path}')

    def test_kako(self,SAVE=False):
        '''
        测试卡口车牌图片
        :return:
        '''
        inf = inference()

        img_glob='/data1/xiancai/PLATE_DATA/kakou/plate/*/*'
        save_dir='/data1/xiancai/PLATE_DATA/other/test_05_23/'

        ls=glob.glob(img_glob)
        acc=0.0
        infor={'err':[],'pre':[]}
        for ind, i in enumerate(ls):
            pre=inf.infer(i)
            plate_str=''.join(inf.CHARS[pre])
            filename=i.split('/')[-1]
            lab=filename[1:8] if filename[8] in ['.','('] else filename[1:9]
            if plate_str == lab:
                acc+=1
            else:
                infor['err'].append(i)
                infor['pre'].append(plate_str)
            print(f'{ind}/{len(ls)} lab/pre: {lab} {plate_str}')
        acc = acc/len(ls)
        print(acc)
        # save
        if SAVE:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for ind,i in enumerate(infor['err']):
                filename=i.split('/')[-1]
                save_path =save_dir+filename+infor['pre'][ind]+'.jpg'
                shutil.copy(i,save_path)


if __name__=='__main__':
    test().test_kako(SAVE=True)

