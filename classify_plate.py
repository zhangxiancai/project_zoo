'''
#!/usr/bin/env python3
测试车牌识别模型 onnx
'''
import glob
import os
import random
import re
import shutil
import time

import onnxruntime as ort
import cv2
import numpy as np

class inference_plate:
    '''
    车牌识别推理
    '''
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

    # init
    def __init__(self,model_path):
        self.model_path = model_path
        self.ses = ort.InferenceSession(model_path)
        print(f'plate_classify model: {self.model_path}')

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

    def preproccess(self,img_path):
        img=cv2.imread(img_path) # to h,w,c
        img=cv2.resize(img,(94,24)) # w=94,h=24
        # img=img[...,::-1] # to RGB
        img=img.transpose(2, 0, 1) # to 3*h*w
        img=img[None,...] # to 1*3*w*h
        img = np.ascontiguousarray(img).astype(np.float32) # to float
        img=(img-127.5)/128 # to [-1,1]
        return img

    def classify_plate(self,img_path):
        '''
        车牌识别接口
        :param img_path:
        :return: 1*~ list int
        '''
        img=cv2.imread(img_path) if isinstance(img_path,str) else img_path # to h,w,c
        img=cv2.resize(img,(94,24)) # w=94,h=24
        # img=img[...,::-1] # to RGB
        img=img.transpose(2, 0, 1) # to 3*h*w
        img=img[None,...] # to 1*3*w*h
        img = np.ascontiguousarray(img).astype(np.float32) # to float
        img=(img-127.5)/128 # to [-1,1]
        # return img # debug
        pre=self.ses.run(None,{'data':img})[0]
        # print(pre.shape)

        pre=np.mean(pre,axis=2)
        # pre=np.max(pre,axis=1)
        pre=np.argmax(pre,axis=1)
        # res_lab = CHARS[pre]
        res=self.remove_repeate_blank_label(pre) #删除重复和占位符 pre：['苏' 'E' '-' '-' '-' '6' '-' '-' '3' '-' '-' '7' '-' '-' 'A' '-' '-' 'F']
        # res_lab=CHARS[res]
        # print(res_lab)
        return res


class test(inference_plate):
    '''
    各种场景的测试
    '''

    def __init__(self,
                 model_path = '',
                 imgs_path='',
                 SAVE = False,
                 # CHARS_NUMB_ACC = True,
                 # CHARS_ACC=None,
                 ACC=0,
                 GREEN = False,
                 WAN=False):
        super(test, self).__init__(model_path)
        self.imgs_path = imgs_path
        self.SAVE = SAVE
        # self.CHARS_NUMB_ACC = CHARS_NUMB_ACC  # 中文字符及数字全对为正确
        # self.CHARS_ACC=CHARS_ACC
        self.ACC=ACC
        # imgs_path为txt时
        self.GREEN = GREEN  # 只测试绿牌
        self.WAN=WAN # 测试特定省牌


    def read_label_from_filename(self,filename):
        # 从不同格式文件名读取tru_lab
        tru_lab = re.split('-|_', filename)[-1][:-4].strip()  # ????????deepcamdata clean
        # tru_lab = filename[:-4] # format:川X90621.jpg

        return tru_lab

    def info(self):

        print(f'test_data:{self.imgs_path}\ntest_model:{self.model_path}\nSAVE:{self.SAVE}\nACC:{self.ACC}\nGREEN:{self.GREEN}')

    def test(self):
        '''
        执行测试
        :return:
        '''
        self.info()

        # load data
        # DIR_TAG = False
        if os.path.isdir(self.imgs_path):
            ls = glob.glob(self.imgs_path+'*')
            # DIR_TAG = True
        else:
            with open(self.imgs_path, 'r') as f:
                ls = list(map(lambda x: x.strip(), f.readlines()))
                ls_c = []
                # 只测试绿牌
                if self.GREEN:
                    for i in ls:
                        if i.split('/')[-2] == 'CCPD2020_ccpd_green':
                            ls_c.append(i)
                    ls = ls_c
                    # ls = ls[:1000]
                ls_c = []
                if self.WAN:
                    for i in ls:
                        if not self.read_label_from_filename(i)[0] in ['皖']:
                            ls_c.append(i)
                    ls=ls_c

        # test
        errs = []
        tag = 0
        res_pro = {}  # {省份: [count,err]}
        random.shuffle(ls)
        # ls = ls[:1000]
        for ind, i in enumerate(ls): # abs_path
            # if not DIR_TAG:
            #     ii = i.split('/')[-1]
            #     imgs_path = i[:len(i) - len(ii)]
            #     i = ii
            # else:
            #     imgs_path = self.imgs_path
            #     # print(imgs_path)
            #     # print(i)
            t0 = time.time()
            pre = self.classify_plate(i)  # classify
            t1 = time.time()

            # count
            pre_lab = ''.join(self.CHARS[pre])
            tru_lab = self.read_label_from_filename(i)
            if not res_pro.get(tru_lab[0]):
                res_pro[tru_lab[0]] = [1,0] # {省份: [count,err]}
            else:
                res_pro[tru_lab[0]][0] += 1

            if self.ACC==0:  # 精度：中文字符及数字全对
                if pre_lab != tru_lab:
                    res_pro[tru_lab[0]][1] += 1  #
                    errs.append(( i, pre_lab))
                    tag += 1
            elif self.ACC==1: # 仅中文字符正确
                if pre_lab[0] != tru_lab[0]:
                    errs.append((i, pre_lab))
                    tag += 1
            elif self.ACC==2:  # 数字字母全对
                if pre_lab[1:] != tru_lab[1:]:
                    errs.append((i, pre_lab))
                    tag += 1
            print(f'{ind}/{len(ls)},inference:{t1 - t0}s')
        acc = 1 - tag / len(ls)
        print(f'acc:{acc}  {tag}/{len(ls)}')
        # {省份: [count,err,acc]}
        for k,v in res_pro.items():
            acc_pro=round(1 - v[1] / v[0], 4)
            v.append(acc_pro)

        res_pro_sorted = dict(sorted(res_pro.items(), key=lambda x: x[1][0], reverse=True))  # 将res_pro按照v[0]降序排列
        print(f'省份: [total,err,acc]: {res_pro_sorted}')

        # save
        if self.SAVE:
            sub_dir = '_'.join(self.model_path.split('/')[-2:]) + '--' + '_'.join(
                self.imgs_path.split('/')[-2:]) + '--' + f'acc{round(acc, 4)}' # model--img--acc/
            err_path = f'/data1/xiancai/PLATE_DATA/res_classify_err/{sub_dir}/'
            if not os.path.exists(err_path):
                os.mkdir(err_path)
            for img_path,  pre_lab in errs:
                shutil.copy(img_path , err_path + img_path.split('/')[-1][:-4] + '-' + 'pre' + pre_lab + '.jpg')
                print(f'err img saved to {err_path}')


# def select_condition(ab_path):






if __name__=='__main__':
    ''' 
    测试车牌识别模型onnx，计算精度
    测试一个文件夹的图片或val.txt，  图片名称格式 ~-川JK0707.jpg   or ~_川JK0707.jpg
    错误图片保存至err_path
    '''

    # model_path = '/home/xiancai/LPRNet_Pytorch/LPRNet.onnx'
    # model_path='/home/xiancai/LPRNet_Pytorch/result/2022_01_12/best_adjust_0.9811.onnx'
    # model_path='/home/xiancai/LPRNet_Pytorch/result/test/best_0.9815_gen_change_color.onnx'
    # model_path='/home/xiancai/LPRNet_Pytorch/result/test/best_0.9566_focal_rmsprop_nopretrain_epoch100.onnx'
    # model_path = '/home/xiancai/LPRNet_Pytorch/result/test/best_0.9811_gen_plate.onnx'
    # model_path = '/home/xiancai/LPRNet_Pytorch/result/test/best_0.9814_change_color.onnx'
    # model_path = '/home/xiancai/LPRNet_Pytorch/result/2022_01_10/best_0.9665.onnx'
    # model_path = '/home/xiancai/plate/LPRNet_Pytorch/Result/2022_02_21/best_0.9816_focal_changev2_genv2_sim.onnx'
    model_path='/home/xiancai/plate/LPRNet_Pytorch/Result/2022_05_23/best_0.9826_sim.onnx'

    imgs_path = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
    # imgs_path = '/data1/xiancai/PLATE_DATA/yello_326/plate_clean/'
    # imgs_path = '/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow_plate/'

    SAVE = False
    ACC = 0 # 0: 测试中文字符及数字 1: 仅测试中文字符 2：只测试数字和字母
    # CHARS_NUMB_ACC = False  # 中文字符及数字全对为正确
    # CHARS_ACC=True if not CHARS_NUMB_ACC else None # (当CHARS_NUMB_ACC为false时有效)True：仅测试中文字符

    # imgs_path为txt时有效
    GREEN = False  # True: 只测试绿牌
    WAN=False ## True: 不测试皖牌

    # if ACC==0:
    #     CHARS_NUMB_ACC,CHARS_ACC=True,None
    # if ACC==1:
    #     CHARS_NUMB_ACC, CHARS_ACC = False, True
    # if ACC==2:
    #     CHARS_NUMB_ACC, CHARS_ACC = False, False

    test(model_path=model_path,imgs_path=imgs_path,SAVE=SAVE,ACC=ACC,GREEN=GREEN,WAN=WAN).test()



