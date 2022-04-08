import glob
import os
import random

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms

from PIL import Image, ImageFilter
import rep_utils
import albumentations #数据增强库 numpy


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

    
class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = rep_utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = rep_utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
     
        R = rep_utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])


        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = rep_utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        # img and mat
        if random.random()<-1:
            img = np.random.randint(0,255,(112,112,3),dtype=np.uint8)
            img = Image.fromarray(img)
            pitch,yaw,roll= 100/180*np.pi,100/180*np.pi,100/180*np.pi
        else:
            img = Image.open(os.path.join(
                self.data_dir, self.X_train[index] + self.img_ext)) # h*w*c
            mat_path = os.path.join(
                self.data_dir, self.y_train[index] + self.annot_ext)
            pt2d = rep_utils.get_pt2d_from_mat(mat_path) #
            pose = rep_utils.get_ypr_from_mat(mat_path)  ##We get the pose in radians
            # print(f'pt2d:{pt2d}')
            # print(f'pose{pose}')

            # crop
            # img.save(f'/data1/xiancai/FACE_ANGLE_DATA/test/debug/origin_{index}.jpg')
            img = img.convert(self.image_mode)
            x_min = min(pt2d[0, :])
            y_min = min(pt2d[1, :])
            x_max = max(pt2d[0, :])
            y_max = max(pt2d[1, :])
            k = np.random.random_sample() * 0.2 + 0.2 # k = 0.2 to 0.40
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            # img.save(f'/data1/xiancai/FACE_ANGLE_DATA/test/debug/debug_{index}.jpg')
            # print(index)

            # p y r 弧度
            pitch = pose[0] #
            yaw = pose[1] #
            roll = pose[2] #


        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # convert to lab
        R = rep_utils.get_R(pitch, yaw, roll)#+ noise

        if self.transform is not None:
            img = self.transform(img)

        return img,  torch.FloatTensor(R),[], self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

def getDataset(dataset, data_dir, filename_list, transformations, train_mode = True):
    if dataset == 'Pose_300W_LP':
            pose_dataset = Pose_300W_LP(
                data_dir, filename_list, transformations)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000(
            data_dir, filename_list, transformations)
    elif dataset == 'BIWI':
        pose_dataset = BIWI(
            data_dir, filename_list, transformations, train_mode = train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(
            data_dir, filename_list, transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(
            data_dir, filename_list, transformations)
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset


class UserDateset(Dataset):
    '''
    训练用
    '''
    def __init__(self,txts=[],imgsize=112):
        self.items = []
        print(f'txts: {txts}')
        for txt in txts:
            with open(txt) as f:
                self.items.extend(map(lambda x:x.strip(),f.readlines())) # [path  p  y  r] 角度单位: degree

        # random.shuffle(self.items)
        self.augment = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(imgsize*1.07)),
            torchvision.transforms.RandomCrop(imgsize)
            ])
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):

        # img and p,y,r
        item=self.items[index]
        its = item.split('  ') # 两个空格
        img_path, p,y,r=its[0],float(its[1]),float(its[2]),float(its[3]) #
        img = Image.open(img_path) # rgb
        # img.save('/data1/xiancai/FACE_ANGLE_DATA/other/debug/debug0.jpg')

        # augment
        if '300W_LP' in img_path:
            img = self.crop(img,mat_path=img_path[:-3]+'mat')
        img = self.augment(img)
        # img.save('/data1/xiancai/FACE_ANGLE_DATA/other/debug/debug.jpg')

        # convert
        img = self.normalize(img) # to tensor, to normalize
        p,y,r = p / 180 * np.pi, y / 180 * np.pi, r / 180 * np.pi # degree to radian
        R = rep_utils.get_R(p,y,r)
        R = torch.FloatTensor(R) # to tensor

        return img, R, [], []

    def __len__(self):
        return len(self.items)

    def crop(self,img,mat_path):
        '''
        300W_LP  根据mat文件里的关键点进行随机crop
        :param img:
        :param mat_path:
        :return:
        '''
        pt2d = rep_utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])
        k = np.random.random_sample() * 0.2 + 0.2  # k = 0.2 to 0.40
        # k=0.3
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        return img

class UserTestDataset(Dataset):
    '''
    测试用 质量不好0, 质量好1
    '''
    def __init__(self,txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/test.txt']):
        self.items = []
        for txt in txts:
            with open(txt) as f:
                self.items.extend(map(lambda x: x.strip(), f.readlines())) # [path cls]

        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        its=self.items[index].split(' ')
        img_path, lab = its[0], int(its[1])
        img = cv2.imread(img_path)
        img = img[...,::-1] # to rgb
        # cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/other/debug/d1.jpg',img)

        # crop
        h, w = img.shape[:2]  # 原图检测框padding 0.5
        x1 = int(w / 2 * 0.35) # to padding 0.15
        x2 = w - x1
        y1 = int(h / 2 * 0.35)
        y2 = h - y1
        img = img[y1:y2, x1:x2, :]
        # img = cv2.resize(img, (244,244))
        # cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/other/debug/d2.jpg', img)
        img =Image.fromarray(img)
        img = self.normalize(img)
        return img, lab

    def __len__(self):
        return len(self.items)


if __name__=='__main__':
    # r=albumentations.CenterCrop(height=112,width=112,p=1)
    # img=cv2.imread('/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/AFW/AFW_1051618982_1_0.jpg')
    # img=r(image=img)['image']
    # cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/other/debug/aug.jpg',img)

    # from torchvision import transforms
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225])
    # transformations = transforms.Compose([transforms.Resize(int(112*1.07)),
    #                                       transforms.RandomCrop(112),
    #                                       transforms.ToTensor(),
    #                                       normalize])
    # lp_dataset = Pose_300W_LP(data_dir='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP',
    #                             filename_path='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/files.txt',
    #                             transform=transformations)
    # img,R,_,_ = lp_dataset[0]

    # # user_dataset = UserDateset(txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/train.txt'])
    # user_dataset = UserDateset(txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/train.txt'])
    # img_u,R_u,_,_ = user_dataset[6]
    # pass

    user_dataset = UserDateset(txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/train2_clean.txt','/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/train2_clean.txt'])
    for i in user_dataset:
        pass
