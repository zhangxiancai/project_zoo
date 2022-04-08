import glob
import shutil

from model import SixDRepNet
import math
import re
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_unchecked

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import datasets
import rep_utils
# import matplotlib
#
# matplotlib.use('TkAgg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000', type=str)
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/files.txt', type=str)  # datasets/BIWI_noTrack.npz
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.',
                        default='AFLW2000', type=str)

    args = parser.parse_args()
    return args


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def ttest(model,test_loader,args):
    # 训练时测试
    model.eval()
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0
    with torch.no_grad():
        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).cuda(args.gpu_id)
            total += cont_labels.size(0)

            # gt
            # gt matrix
            R_gt = r_label
            # gt euler
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            # pre
            R_pred = model(images)
            euler = rep_utils.compute_euler_angles_from_rotation_matrices(
                R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            # count
            R_pred = R_pred.cpu()
            v1_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180 / np.pi)
            v2_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180 / np.pi)
            v3_err += torch.sum(torch.acos(torch.clamp(
                torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180 / np.pi)

            pitch_error += torch.sum(torch.min(
                torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                    p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg),
                             torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(
                torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                    y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg),
                             torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(
                torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                    r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg),
                             torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])


        mae=(yaw_error + pitch_error + roll_error) / (total * 3)
        print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            mae))
        model.train()

        return round(mae.item(),4)

        # print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
        #     v1_err / total, v2_err / total, v3_err / total,
        #     (v1_err + v2_err + v3_err) / (total * 3)))

def ttest_user(model,test_loader,device = torch.device('cuda:0'),threshold=0.75):
    '''
    测试自定义数据集（用于训练时测试） (两类：质量好1，质量不好0)
    :param model:
    :param test_loader:
    :param args:
    :return:
    '''
    model.eval()
    acc=0
    with torch.no_grad():
        for ind,(imgs, labs) in enumerate(test_loader):

            # pre
            imgs = imgs.to(device)
            # print(imgs)
            R_pred = model(imgs)
            euler = rep_utils.compute_euler_angles_from_rotation_matrices(R_pred,use_gpu=False) * 180 / np.pi
            euler = euler.numpy()
            p_pred_deg = euler[:, 0].reshape(-1,1)
            y_pred_deg = euler[:, 1].reshape(-1,1)
            #
            # print(p_pred_deg.shape)
            # print(p_pred_deg)
            p_ratio, y_ratio = np.maximum(1 - np.abs(p_pred_deg) / 100,0.001), np.maximum(1- np.abs(y_pred_deg) / 100,0.001)

            q = np.round(2 * p_ratio * y_ratio / (p_ratio + y_ratio), 2)
            # print(q)
            pres = (q>threshold) # True:正脸，False:侧脸
            # labs = (labs==1)
            labs=labs.numpy().reshape(-1,1)
            acc+=np.sum(pres==labs)

            # print(f'{ind}/{len(test_loader.dataset)}: path:  p,y,r:{int(p_pred_deg)},{int(y_pred_deg)}  q:{round(q.item(), 2)}')
        acc=acc/len(test_loader.dataset)
        print(f'user_acc:{acc},threshold:{threshold}')


def ttest_user_offline(threshold=0.8):
    '''
    使用现场数据集测试（训练完成后）
    :param threshold:
    :return:
    '''
    SAVE=False
    import classify_pt
    infer=classify_pt.inference()
    test_txt='/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/test.txt'
    acc=0.0
    items=[]
    with open(test_txt) as f:
        items.extend(map(lambda x: x.strip().split(' '), f.readlines())) # [img_path lab] 质量不好(大侧脸等)0, 质量好1
    for ind,(img_path,lab) in enumerate(items):
        # padding 0.5 to 0.15
        img = cv2.imread(img_path)
        h, w = img.shape[:2]  # padding 0.5
        x1 = int(w / 2 * 0.35) # to padding 0.15
        x2 = w - x1
        y1 = int(h / 2 * 0.35)
        y2 = h - y1
        img = img[y1:y2, x1:x2, :]

        p,y,r = infer.classify(img)

        p_ratio, y_ratio = max(1 - abs(p) / 100, 0.001), max(1 - abs(y) / 100, 0.001)
        q = 2 * p_ratio * y_ratio / (p_ratio + y_ratio)
        # q =0.8*y_ratio+0.2*p_ratio
        pre= '0' if q<threshold else '1'
        print(f'{ind}/{len(items)}: path:{img_path}  p,y,r:{int(p)},{int(y)},{int(r)}  q:{round(q,2)}')

        if pre == lab:
            acc = acc+1
        else:
            # save error
            if SAVE:
                err_name=f'pre{pre}_'+img_path.split('/')[-1]
                shutil.copy(img_path,f'/data1/xiancai/FACE_ANGLE_DATA/other/err_scene1_v1/{err_name}')
        # draw and save
        if SAVE:
            text=f'{round(q,2)}'
            # img = cv2.resize(img,img.shape[:2]*2)
            cv2.putText(img, text, (5, 15), 0, 0.4, [0, 0, 255] if q<0.7 else [0,255,0], thickness=2)
            save_name=img_path.split('/')[-1]
            cv2.imwrite(f'/data1/xiancai/FACE_ANGLE_DATA/other/res_scene1_v1/{save_name}',img)

    acc=acc/len(items)
    print(f'acc:{acc},threshold:{threshold}')


def ttest_img_mult():
    '''
    测试多个图片并画图保存
    :return:
    '''
    imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/origin_imgs_labs_a0_clean/*'
    save_dir = '/data1/xiancai/FACE_ANGLE_DATA/other/res_04_02/'
    ls = glob.glob((imgs_glob))
    import classify_pt
    infer=classify_pt.inference()
    for i in ls:
        img = cv2.imread(i)
        p, y, r = infer.classify(img)
        # quality
        p_ratio, y_ratio = max(1 - abs(p) / 100, 0.001), max(1 - abs(y) / 100, 0.001)
        q = 2 * p_ratio * y_ratio / (p_ratio + y_ratio)
        # draw and save
        text = f'{round(q, 2)}'
        cv2.putText(img, text, (20, 20), 0, 0.6, [0, 0, 255] if q < 0.7 else [0, 255, 0], thickness=2)
        save_name = i.split('/')[-1]
        cv2.imwrite(f'{save_dir}{save_name}', img)


if __name__ == '__main__':

    ttest_user_offline()


    # device = torch.device('cuda:0')
    # user_test_dataset = datasets.UserTestDataset(
    #     txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/test.txt'])  # 第二个测试集
    # user_test_loader = torch.utils.data.DataLoader(
    #     dataset=user_test_dataset,
    #     shuffle=False,
    #     batch_size=1,
    #     num_workers=2)
    # # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_07/RepVGG-A0s_epoch_180_mae8.0871_transfer.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_07/RepVGG-A0s_epoch_180_mae8.2222.pth'
    # model = SixDRepNet(backbone_name='RepVGG-A0s',
    #                    backbone_file='',
    #                    deploy=True,
    #                    pretrained=False,use_gpu=True).to(device)
    # model.load_state_dict(torch.load(pt_path))
    #
    # ttest_user(model,user_test_loader)



