from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
import albumentations as A

import re

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '警','挂',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ] # 双层车牌

CHARS_DICT = {char:i for i, char in enumerate(CHARS)} # {char:id}


transform = A.Compose([
    A.ToGray(p=0.1),
    A.ChannelShuffle(p=0.02),

    A.HueSaturationValue(p=0.1),
    A.MotionBlur(blur_limit=[5, 12], p=0.1),
    A.RandomBrightness(p=0.1),
    A.RandomBrightness(limit=(0.7,0.8),p=0.01),
    A.RandomBrightness(limit=(-0.8,-0.7),p=0.01),

    A.ISONoise(p=0.05),
    A.MultiplicativeNoise(p=0.05),
    A.RGBShift(p=0.05),
    A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1)
])

DEBUG=False

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, augment=False):
        # 读取train.txt 至 self.img_paths
        self.img_dir = img_dir
        self.img_paths = []
        img_paths=[]
        for img_di in img_dir:
            with open(img_di,'r') as f:
                img_paths+=list(map(lambda x:x.strip(),f.readlines()))
        self.img_paths = img_paths
        random.shuffle(self.img_paths)  # 注意
        if DEBUG:
            self.img_paths=img_paths[:1000]

        self.augment = augment
        self.img_size = imgSize # imgSize=opt.img_size 94*24
        self.lpr_max_len = lpr_max_len
        self.ind=0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]

        # lab
        basename = os.path.basename(filename) # basename=''0301245210728-89_87-233&550_615&629-614&624_236&645_239&555_617&534-0_0_7_28_30_30_32-121-182-皖AH4668.jpg''
        imgname, suffix = os.path.splitext(basename)
        # imgname = imgname.split("-")[0].split("_")[0]
        imgname=re.split('_|-',imgname)[-1].strip() ##
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])


        # Img augment
        Image = cv2.imread(filename)  # 载入图片

        if self.augment:
            Image = transform(image=Image)["image"]
            if self.ind<1000:
                cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/test_04_19/augment/{self.ind}.jpg',Image)
                self.ind+=1

        # 归一化
        Image = self.transform(Image)

        # result
        if DEBUG:
            return Image, label, len(label),filename
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125 # 1/128
        img = np.transpose(img, (2, 0, 1)) # 注意 to 3*h*w

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
