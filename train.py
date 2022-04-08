import sys
import os
import argparse
import time

import numpy as np
import pylab as pl

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from model import SixDRepNet, SixDRepNet2, Resnet18_face_angle
import datasets
from loss import GeodesicLoss

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
        parser.add_argument('--lr', dest='lr', help='Base learning rate.',default=0.0000025, type=float) # 0.00001
        parser.add_argument('--img_size', dest='img_size', help='Img size.', default=112, type=int) # 图片大小

        parser.add_argument('--dataset', dest='dataset', help='Dataset type.',default='Pose_300W_LP', type=str) #Pose_300W_LP
        parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP', type=str)#BIWI_70_30_train.npz
        parser.add_argument('--filename_list', dest='filename_list',help='Path to text file containing relative paths for every example.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt

        # test data
        parser.add_argument('--test_data_dir',dest='test_data_dir', help='Directory path for data.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/', type=str)
        parser.add_argument('--test_filename_list',dest='test_filename_list',help='Path to text file containing relative paths for every example.',default='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/files.txt', type=str)
        parser.add_argument('--test_dataset',dest='test_dataset', help='Dataset type.',default='AFLW2000', type=str)

        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
        parser.add_argument('--output_string', dest='output_string',help='String appended to output snapshots.', default='', type=str)
        parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='/home/xiancai/face_angle/6DRepNet/results/2022_03_16/_epoch_30.tar', type=str)
        args = parser.parse_args()
        return args

class train_a0:
    '''
    训练A0模型
    '''

    args = parse_args()
    # CUDA_VISIBLE_DEVICES='1,2,3,4'
    backbone_name='RepVGG-A0'
    backbone_file='/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0-train.pth'
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
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA_VISIBLE_DEVICES
        gpu = self.args.gpu_id

        if not os.path.exists('output/snapshots'):
            os.makedirs('output/snapshots')

        summary_name = '{}_{}_bs{}'.format(
            'SixDRepNet', int(time.time()), self.args.batch_size)

        if not os.path.exists('output/snapshots/{}'.format(summary_name)):
            os.makedirs('output/snapshots/{}'.format(summary_name))

        # model
        # model = SixDRepNet(backbone_name='RepVGG-B1g2',
        #                     backbone_file='RepVGG-B1g2-train.pth',
        #                     deploy=False,
        #                     pretrained=True)
        model = SixDRepNet(backbone_name=self.backbone_name,
                            backbone_file=self.backbone_file,
                            deploy=False,
                            pretrained=True)

        if not self.args.snapshot == '':
            saved_state_dict = torch.load(self.args.snapshot)
            model.load_state_dict(saved_state_dict['model_state_dict'])
        model.cuda(gpu)

        # # freeze
        # freeze=['layer0','layer1','layer2','layer3','layer4']
        # for k, v in model.named_parameters():
        #     v.requires_grad = True  # train all layers
        #     if any(x in k for x in freeze):#
        #         print(f'freezing {k}')
        #         v.requires_grad = False

        # data
        print('Loading data.')
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])
        # transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
        #                                       transforms.RandomCrop(self.args.img_size),
        #                                       transforms.ToTensor(),
        #                                       normalize])
        # pose_dataset = datasets.getDataset(
        #     self.args.dataset, self.args.data_dir, self.args.filename_list, transformations)

        pose_dataset = datasets.UserDateset(txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/train.txt',
                                                  '/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/train2_clean.txt',
                                                  '/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/train2_clean.txt'],
                                            imgsize=self.args.img_size)
        train_loader = torch.utils.data.DataLoader(
            dataset=pose_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)
        # testing data
        # test_transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
        #                                       transforms.CenterCrop(
        #                                           self.args.img_size), transforms.ToTensor(),
        #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                            std=[0.229, 0.224, 0.225])])
        test_transformations = transforms.Compose([transforms.Resize(int(self.args.img_size*1.07)),
                                              transforms.CenterCrop(
                                                  self.args.img_size), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.225, 0.225, 0.225])])
        test_pose_dataset = datasets.getDataset(
            self.args.test_dataset, self.args.test_data_dir, self.args.test_filename_list, test_transformations, train_mode=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_pose_dataset,
            batch_size=self.args.batch_size,
            num_workers=2)

        user_test_dataset = datasets.UserTestDataset(txts=['/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/test.txt']) # 第二个测试集
        user_test_loader = torch.utils.data.DataLoader(
            dataset=user_test_dataset,
            batch_size=self.args.batch_size,
            num_workers=2)


        # loss, opt, lr
        crit = GeodesicLoss().cuda(gpu)  # torch.nn.MSELoss().cuda(gpu)
        optimizer = torch.optim.Adam([
            {'params': self.get_ignored_params(model), 'lr': 0},
            {'params': self.get_non_ignored_params(lays=[model.layer1,model.layer2]), 'lr': self.args.lr},
            {'params': self.get_non_ignored_params(lays=[model.layer3, model.layer4]), 'lr': self.args.lr * 10},
            {'params': self.get_fc_params(lays=[model.linear_reg, model.se]), 'lr': self.args.lr * 10}
        ], lr=self.args.lr)
        # optimizer = torch.optim.Adam([
        #     {'params': self.get_ignored_params(model), 'lr': 0},
        #     {'params': self.get_non_ignored_params(lays=[model.layer1,model.layer2]), 'lr': 0},
        #     {'params': self.get_non_ignored_params(lays=[model.layer3, model.layer4]), 'lr': 0},
        #     {'params': self.get_fc_params(lays=[model.linear_reg, model.se]), 'lr': self.args.lr * 10}
        # ], lr=self.args.lr)
        print(f'opt: {optimizer}')

        # if not self.args.snapshot == '':
        #     optimizer.load_state_dict(saved_state_dict['optimizer_state_dict']) # 载入预训练模型的当前学习率
        # milestones = np.arange(num_epochs)
        milestones = [90, 150]
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
                pred_mat = model(images)

                # Calc loss
                loss = crit(gt_mat.cuda(gpu), pred_mat)

                optimizer.zero_grad()
                loss.backward()
                # wg=model.se.down.weight.grad
                # bg=model.se.down.bias.grad
                # print(f'wg{wg},bg{bg}')

                optimizer.step()

                loss_sum += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] lr %.8f Loss: '
                          '%.6f' % (
                              epoch + 1,
                              num_epochs,
                              i + 1,
                              len(pose_dataset) // batch_size,
                              scheduler.get_last_lr()[1],
                              loss.item(),
                          )
                          )

            scheduler.step()

            # test and save
            if epoch % 4 == 0 or epoch == num_epochs - 1:
                import test
                mae = test.ttest(model,test_loader,self.args)
                test.ttest_user(model,user_test_loader) # 自定义数据集测试
                if mae<best_mae:
                    best_mae = mae
                    best_epoch=epoch
                print('Taking snapshot...',
                      torch.save({
                          'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                      }, 'output/snapshots/' + summary_name + '/' + self.backbone_name +
                          '_epoch_' + str(epoch+1) + f'_mae{mae}.tar')
                      )
                print('output/snapshots/' + summary_name + '/' )

                # save to plt
                self.plot.append(loss.item(),mae,scheduler.get_last_lr()[1])

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