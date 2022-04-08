'''
训练脚本 蒸馏学习
'''
import sys
import os
import argparse
import time

import numpy as np
import pylab as pl

import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn

from model import SixDRepNet, SixDRepNet2, Resnet18_face_angle
import datasets
from loss import GeodesicLoss,Distilling_loss1

import torch.utils.model_zoo as model_zoo
import torchvision
import rep_utils


import matplotlib.pyplot as plt

class plot_lml:
    '''
    画训练过程中的loss mae
    '''
    Loss=[]
    Mae=[]
    Lr=[]
    save_path=f'/home/xiancai/face_angle/6DRepNet/log/'
    prefix = f'{int(time.time())}_'
    def append(self,los,ma,lr):
        self.Loss.append(los)
        self.Mae.append(ma)
        self.Lr.append(lr)

    def plot(self):
        lab=['loss','mae','lr']
        lml=np.array((self.Loss,self.Mae,self.Lr))
        for i in range(3):
            plt.plot(lml[i])
            # plt.xlabel('Epoch')
            plt.ylabel(lab[i])
            plt.savefig(self.save_path+self.prefix+lab[i]+'.png')
            plt.close()
        print(f' pngs (loss mae lr) saved to {self.save_path}')

def parse_args():

        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Head pose estimation using the 6DRepNet.')


        # parser.add_argument(
        #     '--CUDA_VISIBLE_DEVICES',
        #     default='0,1,2,3,4')

        parser.add_argument('--num_epochs', dest='num_epochs',help='Maximum number of training epochs.',default=180, type=int)
        parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',default=64, type=int)
        parser.add_argument('--lr', dest='lr', help='Base learning rate.',default=0.00001, type=float) # 0.00001
        parser.add_argument('--img_size', dest='img_size', help='Img size.', default=224, type=int) # teacher图片大小

        parser.add_argument('--dataset', dest='dataset', help='Dataset type.',default='Pose_300W_LP', type=str) #Pose_300W_LP
        parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP', type=str)#BIWI_70_30_train.npz
        parser.add_argument('--filename_list', dest='filename_list',help='Path to text file containing relative paths for every example.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt


        # test data
        parser.add_argument('--test_data_dir',dest='test_data_dir', help='Directory path for data.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/', type=str)
        parser.add_argument('--test_filename_list',dest='test_filename_list',help='Path to text file containing relative paths for every example.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/files.txt', type=str)
        parser.add_argument('--test_dataset',dest='test_dataset', help='Dataset type.',default='AFLW2000', type=str)

        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
        parser.add_argument('--output_string', dest='output_string',help='String appended to output snapshots.', default='', type=str)
        parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='', type=str)
        args = parser.parse_args()
        return args

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
            R_pred,_ = model(images)
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



class train_a0:
    '''
    训练A0模型
    '''

    args = parse_args()
    CUDA_VISIBLE_DEVICES='1,2,3,4'
    backbone_name='RepVGG-A0s-stu'
    backbone_file='/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0s-stu-train.pth'
    plot=plot_lml()

    # def get_ignored_params(self, model):
    #     b = [model.layer0]
    #     # b = [model.conv1, model.bn1, model.fc_finetune]
    #     for i in range(len(b)):
    #         for module_name, module in b[i].named_modules():
    #             if 'bn' in module_name:
    #                 module.eval()
    #             for name, param in module.named_parameters():
    #                 yield param

    # def get_non_ignored_params(self, model):
    #     b = [model.layer1, model.layer2, model.layer3, model.layer4]
    #     for i in range(len(b)):
    #         for module_name, module in b[i].named_modules():
    #             if 'bn' in module_name:
    #                 module.eval()
    #             for name, param in module.named_parameters():
    #                 yield param
    #
    # def get_fc_params(self, model):
    #     b = [model.linear_reg]
    #     for i in range(len(b)):
    #         for module_name, module in b[i].named_modules():
    #             for name, param in module.named_parameters():
    #                 yield param

    def get_ignored_params(self, model):
        b = [model.layer0]
        # b = [model.conv1, model.bn1, model.fc_finetune]
        for i in range(len(b)):
            for module_name, module in b[i].named_modules():
                if 'bn' in module_name:
                    module.eval()
            for name, param in b[i].named_parameters():
                yield param

    def get_non_ignored_params(self, lays=[]):
        # b = [model.layer1, model.layer2, model.layer3, model.layer4]
        b = lays
        for i in range(len(b)):
            for module_name, module in b[i].named_modules():
                if 'bn' in module_name:
                    module.eval()
            for name, param in b[i].named_parameters():
                yield param


    def get_fc_params(self,lays=[]):
        # b = [model.linear_reg]
        b = lays
        for i in range(len(b)):
            for name, param in b[i].named_parameters():
                yield param


    def load_filtered_state_dict(self, model, snapshot):
        # By user apaszke from discuss.pytorch.org
        model_dict = model.state_dict()
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)

    def train(self):
        '''
        a0
        :return:
        '''
        print(self.args)
        cudnn.enabled = True
        num_epochs = self.args.num_epochs
        batch_size = self.args.batch_size
        os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA_VISIBLE_DEVICES
        gpu = self.args.gpu_id

        if not os.path.exists('output/snapshots'):
            os.makedirs('output/snapshots')

        summary_name = '{}_{}_bs{}'.format(
            'SixDRepNet', int(time.time()), self.args.batch_size)

        if not os.path.exists('output/snapshots/{}'.format(summary_name)):
            os.makedirs('output/snapshots/{}'.format(summary_name))

        # model
        student = SixDRepNet(backbone_name=self.backbone_name,
                            backbone_file=self.backbone_file,
                            deploy=False,
                            pretrained=True,student=True)

        if not self.args.snapshot == '':
            saved_state_dict = torch.load(self.args.snapshot)
            student.load_state_dict(saved_state_dict['model_state_dict'])
        student.cuda(gpu)

        # training data
        print('Loading data.')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.225, 0.225, 0.225])
        # transformations = transforms.Compose([transforms.Resize(240),
        #                                       transforms.RandomCrop(224),
        #                                       transforms.ToTensor(),
        #                                       normalize])
        transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
                                              transforms.RandomCrop(self.args.img_size),
                                              transforms.ToTensor(),
                                              normalize])
        pose_dataset = datasets.getDataset(
            self.args.dataset, self.args.data_dir, self.args.filename_list, transformations)
        train_loader = torch.utils.data.DataLoader(
            dataset=pose_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        # load testing data
        test_transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
                                              transforms.CenterCrop(
                                                  self.args.img_size), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        # test_transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
        #                                       transforms.CenterCrop(
        #                                           self.args.img_size), transforms.ToTensor(),
        #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                            std=[0.225, 0.225, 0.225])])
        test_pose_dataset = datasets.getDataset(
            self.args.test_dataset, self.args.test_data_dir, self.args.test_filename_list, test_transformations, train_mode=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_pose_dataset,
            batch_size=self.args.batch_size,
            num_workers=2)

        # loss, opt, lr
        loss1 = Distilling_loss1().cuda(gpu)
        loss2 = GeodesicLoss().cuda(gpu)  # torch.nn.MSELoss().cuda(gpu)

        optimizer = torch.optim.Adam([
            {'params': self.get_ignored_params(student), 'lr': 0},
            {'params': self.get_non_ignored_params(lays=[student.layer1,student.layer2]), 'lr': self.args.lr},
            {'params': self.get_non_ignored_params(lays=[student.layer3, student.layer4]), 'lr': self.args.lr * 10},
            {'params': self.get_fc_params(lays=[student.linear_reg, student.se]), 'lr': self.args.lr * 10}
        ], lr=self.args.lr)
        # optimizer = torch.optim.Adam([
        #     {'params': self.get_ignored_params(model), 'lr': 0},
        #     {'params': self.get_non_ignored_params(lays=[model.layer1,model.layer2,model.layer3, model.layer4]), 'lr': self.args.lr},
        #     {'params': self.get_fc_params(model), 'lr': self.args.lr * 10}
        # ], lr=self.args.lr)

        if not self.args.snapshot == '':
            optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])
        # milestones = np.arange(num_epochs)
        milestones = [60, 120]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)

        # train and save
        best_mae=1000
        print('Starting training.')
        for epoch in range(num_epochs):
            loss_sum = .0
            iter = 0
            for i, (images, gt_mat, _, _) in enumerate(train_loader):
                iter += 1
                images = torch.Tensor(images).cuda(gpu)

                # Forward pass
                pred_mat,stu_feat = student(images)

                # Calc loss
                l1 = loss1(images,stu_feat)
                l2 = loss2(gt_mat.cuda(gpu), pred_mat)
                loss = l1*0.05+l2

                optimizer.zero_grad()
                loss.backward()
                # wg=model.se.down.weight.grad
                # bg=model.se.down.bias.grad
                # print(f'wg{wg},bg{bg}')

                optimizer.step()

                loss_sum += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] lr %.8f Loss: '
                          '%.6f L1: %.6f L2: %.6f' % (
                              epoch + 1,
                              num_epochs,
                              i + 1,
                              len(pose_dataset) // batch_size,
                              optimizer.get_last_lr(),
                              loss.item(),
                              l1.item(),
                              l2.item()
                          )
                          )

            scheduler.step()

            # test and save
            if epoch % 4 == 0 or epoch == num_epochs - 1:

                mae = ttest(student,test_loader,self.args)
                if mae<best_mae:
                    best_mae = mae
                    best_epoch=epoch
                print('Taking snapshot...',
                      torch.save({
                          'epoch': epoch,
                          'model_state_dict': student.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                      }, 'output/snapshots/' + summary_name + '/' + self.backbone_name +
                          '_epoch_' + str(epoch+1) + f'_mae{mae}.tar')
                      )
                print('output/snapshots/' + summary_name + '/' )

                # save to plt
                self.plot.append(loss.item(),mae,scheduler.get_lr()[1])

        self.plot.plot()
        print(f'best mae:{best_mae} epoch:{best_epoch}')



class train_res18:
    '''
    训练res18模型
    '''
    args = parse_args()
    def train(self):

        cudnn.enabled = True
        num_epochs = self.args.num_epochs
        batch_size = self.args.batch_size
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4'
        gpu = self.args.gpu_id

        if not os.path.exists('output/snapshots'):
            os.makedirs('output/snapshots')

        summary_name = '{}_{}_bs{}'.format(
            'SixDRepNet', int(time.time()), self.args.batch_size)

        if not os.path.exists('output/snapshots/{}'.format(summary_name)):
            os.makedirs('output/snapshots/{}'.format(summary_name))

        model = Resnet18_face_angle()  #
        model.load_state_dict(torch.load('output/snapshots/SixDRepNet_1647663302_bs64/_epoch_60_mae12.7453.pth'))

        model.cuda(gpu)

        # data
        print('Loading data.')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transformations = transforms.Compose([transforms.Resize(120),
                                              transforms.RandomCrop(112),
                                              transforms.ToTensor(),
                                              normalize])
        pose_dataset = datasets.getDataset(
            self.args.dataset, self.args.data_dir, self.args.filename_list, transformations)
        train_loader = torch.utils.data.DataLoader(
            dataset=pose_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        # loss, opt, lr
        crit = GeodesicLoss().cuda(gpu)  # torch.nn.MSELoss().cuda(gpu)
        optimizer = torch.optim.SGD(rep_utils.noBiasDecay(model, lr=0.04, weight_decay=1e-4), momentum=0.9)

        # milestones = np.arange(num_epochs)
        milestones = [10, 20]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)

        # train and save
        print('Starting training.')
        for epoch in range(num_epochs):
            loss_sum = .0
            iter = 0
            for i, (images, gt_mat, _, _) in enumerate(train_loader):
                iter += 1
                images = torch.Tensor(images).cuda(gpu)

                # Forward pass
                pred_mat = model(images)

                # Calc loss
                loss = crit(gt_mat.cuda(gpu), pred_mat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] lr %.4f Loss: '
                          '%.6f' % (
                              epoch + 1,
                              num_epochs,
                              i + 1,
                              len(pose_dataset) // batch_size,
                              optimizer.param_groups[0]['lr'],
                              loss.item(),
                          )
                          )

            scheduler.step()

            # test and save
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                import test
                mae = test.ttest(model)

                save_path = 'output/snapshots/' + summary_name + '/' + self.args.output_string + 'res18_epoch_' + str(
                    epoch + 1) + f'_mae{mae}.pth'
                print(f'saved to {save_path}')
                torch.save(model.state_dict(), save_path)




if __name__ == '__main__':

   train_a0().train()