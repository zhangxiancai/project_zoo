import argparse
import glob
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.NoBiasDecay import noBiasDecay 
from data.dataset import Dataset, DataLoaderX
from utils.net_utils import CosineDecayLR, load_model
#from timm.resnet import resnet18d, resnet26d, resnet26t, resnet34d 
from backbones.resnet import resnet18, resnet26, resnet34 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-num-classes', type=int, default=114, help='num classes')
    parser.add_argument('-batch-size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')#0.04
    parser.add_argument('-epochs', type=int, default=150, help='training epoches')
    parser.add_argument('-warmup', type=int, default=5, help='warm up phase')
    parser.add_argument('-root', type=str, default='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset', help='dataset root dir')
    parser.add_argument('-train-list', type=str, default='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/train.txt', help='dataset train list')
    parser.add_argument('-val-list', type=str, default='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/test.txt', help='dataset val list')
    parser.add_argument('-resume', type=str, default='result/2021_11_28/resnet18-137-best.pth11', help='resume model path')
    args = parser.parse_args()

    print(f'total of gpu:{torch.cuda.device_count()}')#
    print(f'dataset:{args.root}')
    #checkpoint directory
    # checkpoint_path = os.path.join('checkpoints', args.net)
    res_sub_dir=f'exp14_{args.net}_cls{args.num_classes}_bat{args.batch_size}_nogamma_agontra_test' #
    checkpoint_path = os.path.join('checkpoints',res_sub_dir )
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path) #清空结果
    os.makedirs(checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #模型
    if args.net == 'resnet18':
        backbone = resnet18(args.num_classes)
    elif args.net == 'resnet26':
        backbone = resnet26(args.num_classes)
    elif args.net == 'resnet26d':
        backbone = resnet26d()   
    elif args.net == 'resnet26t':
        backbone = resnet26t()   
    elif args.net == 'resnet34':
        backbone = resnet34()   
    elif args.net == 'resnet34d':
        backbone = resnet34d()   
    else:
        print('{} not support!!'.format(args.net))
        exit()

    #backbone.reset_classifier(num_classes=args.num_classes)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'  #debug
    #设备

    if torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device('cpu')
    backbone = backbone.to(device) 
    print(backbone)

    if os.path.exists(args.resume):
        # load_model(backbone, args.resume)
        print(f'load state_dict from {args.resume}')
        backbone.load_state_dict(torch.load(args.resume))
    #损失函数和优化算法
    cross_entropy = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(noBiasDecay(backbone, args.lr, 1e-4), momentum=0.9)
    optimizer = optim.SGD(noBiasDecay(backbone, args.lr, 1e-4), momentum=0.9)

    #数据载入
    valset   = Dataset(root_dir=args.root, data_list=args.val_list, local_rank=0)
    trainset = Dataset(root_dir=args.root, data_list=args.train_list, local_rank=0, augment=True) # 训练集 数据增强
    val_loader   = DataLoaderX(local_rank=0, dataset=valset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)
    train_loader = DataLoaderX(local_rank=0, dataset=trainset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)

    #set up warmup phase learning rate scheduler
    iter_per_epoch = len(train_loader)
    #set up training phase learning rate scheduler
    scheduler = CosineDecayLR(optimizer, T_max = iter_per_epoch * args.epochs, lr_init=args.lr, lr_min=1e-5, warmup=args.warmup * iter_per_epoch)

    best_acc = 0.0
    best_epoch=0#
    for epoch in range(1, args.epochs + 1):
        #training procedure
        backbone.train()
        for batch_index, (images, labels) in enumerate(train_loader):
            n_iter = (epoch - 1) * iter_per_epoch + batch_index + 1
            scheduler.step(n_iter)
            optimizer.zero_grad()
            predicts = backbone(images)
            loss = cross_entropy(predicts, labels)
            loss.backward()
            optimizer.step()
            if batch_index % 50 == 0:
                cur_date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('{}:{} Epoch:{} Iter:{}/{} LR:{:0.4f} Loss:{:0.4f}\t'.format(cur_date, res_sub_dir,epoch, batch_index, iter_per_epoch, optimizer.param_groups[0]['lr'], loss.item()))

        correct = 0
        total_loss = 0
        backbone.eval()
        for images, labels in val_loader:
            predicts = backbone(images)
            _, preds = predicts.max(1)
            correct += preds.eq(labels).sum().float()
            loss = cross_entropy(predicts, labels)
            total_loss += loss.item()
        test_loss = total_loss / len(val_loader)
        acc = correct / len(val_loader.dataset)
        print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
        if epoch > 1 and best_acc < acc:
            torch.save(backbone.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type=f'best_{round(float(acc.cpu().numpy()),4)}'))
            best_acc = acc
            best_epoch=epoch #
            continue
        # torch.save(backbone.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    # log
    print(f'best acc: {best_acc}\nbest epoch: {best_epoch}')#
    best_path = checkpoint_path.format(net=args.net, epoch=best_epoch, type=f'best_{round(float(best_acc.cpu().numpy()),4)}')
    print(f'saved to {best_path}')




