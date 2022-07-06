import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision

import cv2
import sys
import random
import albumentations as A # 数据增强库

DEBUG=False
AUGMENT = A.Compose([
    # A.RandomRotate90(p=0.5),  # 旋转
    A.HorizontalFlip(p=0.5),
    A.HueSaturationValue(p=0.1), #色调，饱和度，值
    # A.RandomGamma(p=0.1), # 亮度增强
    # A.MedianBlur(blur_limit=[1, 7], p=0.1),
    # A.RandomCrop(), # 裁剪
    A.RandomBrightnessContrast(p=0.1), #随机对比度
    A.MotionBlur(blur_limit=[5, 12], p=0.1),
    A.RGBShift(p=0.05),
    A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1),
    A.ToGray(p=0.1),
])

PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Resize((112, 112)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# PREPROCESS = A.Compose([
#     A.ToTensor(),
#     A.Resize((112, 112)),
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

def random_crop(img, min_scale=0.80,p=0.1):
    '''
    随机裁剪，(根据图像尺寸)
    :param img:
    :param min_scale:
    :param p:
    :return:
    '''
    if random.random()<p:
        h,w=img.shape[:2]
        rate = random.random()*(1-min_scale)+min_scale
        img = cv2.resize(img,(int(w/rate),int(h/rate)))
        img = A.RandomCrop(height=h,width=w)(image=img)['image']
    return img

def letterbox(img):
    # letterbox 填充为正方形
    h, w = img.shape[:2]
    new_s = max(w, h)
    dw, dh = new_s - w, new_s - h
    top, bottom, left, right = dh // 2, dh - dh // 2, dw // 2, dw - dw // 2
    img_crop = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_crop

class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(112, 112)):
        self.phase = phase
        self.input_shape = input_shape

        # get paths of imgs from txts
        imgs=[]
        for i in data_list_file:
            with open(os.path.join(i), 'r') as fd:
                imgs.extend(fd.readlines())
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        imgs = [os.path.join(root, img.strip()) for img in imgs] # txt文件里的img所加的路径前缀
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        # if self.phase == 'train':
        #     self.transforms = T.Compose([
        #         T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1), # 随机改变亮度，色调，对比度
        #         T.RandomRotation(20, resample=False, expand=False, center=None), # 随机旋转
        #         T.ToTensor(),
        #         normalize
        #     ])
        # else:
        #     self.transforms = T.Compose([
        #         T.ToTensor(),
        #         normalize
        #     ])


    def __getitem__(self, index):
        # try:
        #     sample = self.imgs[index]
        #     splits = sample.split()
        #     img_path = splits[0]
        #     data = Image.open(img_path)
        #     data = data.convert('RGB')
        #     data = self.rand_augment(data) # resize to 112*112 (p=1）,训练时：水平反转(p=0.1），随机裁剪(p=0.1），转灰度图 （p=0.1）
        #     data = self.transforms(data) # 训练时：随机改变亮度0.1，色调0.1，对比度0.1，随机旋转,归一化 测试时:归一化
        #     label = np.int32(splits[1])
        # except:
        #     print(f'sample:{sample}')
        #     raise
        # return data.float(), label

        sample = self.imgs[index]
        img_path,lab = sample.split(' ')

        img = cv2.imread(img_path)
        if img.shape[0]!=img.shape[1]:
            # print(f'origin shape : w is not equal to h {img.shape}')
            img=letterbox(img) # 填充为正方形
            # if DEBUG:
            #     save_name=img_path.split('/')[-1]
            #     cv2.imwrite(f'/data1/xiancai/FACE_DATA/other/debug/pad/{save_name}',img)

        img = img[...,::-1] # to rgb
        # augment
        if self.phase=='train':
            img=random_crop(img,min_scale=0.8,p=0.1)
            img=AUGMENT(image=img)['image']
        # preprocess
        img=PREPROCESS(Image.fromarray(img))
        return img,int(lab)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # dataset = Dataset(root='./',
    #                   data_list_file='./maskface_test_list.txt',
    #                   phase='train',
    #                   input_shape=(3, 64, 64))
    #
    # trainloader = data.DataLoader(dataset, batch_size=32)
    # for i, (data, label) in enumerate(trainloader):
    #     # imgs, labels = data
    #     # print imgs.numpy().shape
    #     # print data.cpu().numpy()
    #     # if i == 0:
    #     img = torchvision.utils.make_grid(data).numpy()
    #     # print img.shape
    #     # print label.shape
    #     # chw -> hwc
    #     img = np.transpose(img, (1, 2, 0))
    #     #img *= np.array([0.229, 0.224, 0.225])
    #     #img += np.array([0.485, 0.456, 0.406])
    #     img += np.array([1, 1, 1])
    #     img *= 127.5
    #     img = img.astype(np.uint8)
    #     img = img[:, :, [2, 1, 0]]
    #
    #     cv2.imshow('img', img)
    #     cv2.waitKey()
    #     # break
    #     # dst.decode_segmap(labels.numpy()[0], plot=True)
    img=cv2.imread('/data1/xiancai/FACE_DATA/wiki/wiki_clean/00/0_10110600_1985-09-17_2012.jpg')
    cv2.imwrite('/data1/xiancai/FACE_DATA/other/debug/1_orin.jpg', img)
    # img=A.RandomCrop(height=212,width=212)(image=img)['image']
    img=random_crop(img,min_scale=0.8)
    cv2.imwrite('/data1/xiancai/FACE_DATA/other/debug/1_crop2.jpg',img)
