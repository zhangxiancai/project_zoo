# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
单机多gpu：dataparallel
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import LPRNet, LPRNet_Double
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
import argparse
import torch
import time
import os

from optim.lars import Lars
from utils.NoBiasDecay import noBiasDecay
from utils.net_utils import CosineDecayLR, load_model

import torch.optim

from loss.focal_ctc import FocalCTCLoss # 结合ctc和focal




def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=100, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/plate_double_generate_train.txt"], help='the train images path')
    parser.add_argument('--test_img_dirs', default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/plate_double_generate_val.txt"], help='the test images path')

    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.01, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=16, help='training batch size.')
    parser.add_argument('--test_batch_size', default=16, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=500, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=500, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    # parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')


    args = parser.parse_args()

    return args



def train():

    # init
    args = get_parser()
    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0
    # save
    import time
    import os
    args.save_folder=f'weights/{str(int(time.time()))}/' # the path to save model
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # model
    # lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet = LPRNet_Double(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 0号不用
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("MODEL: Successful to build network!")
    # weights
    if args.pretrained_model:
        #lprnet.load_state_dict(torch.load(args.pretrained_model))
        load_model(lprnet, args.pretrained_model)
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")
    if torch.cuda.device_count() > 1:
        lprnet = DataParallel(lprnet)  ##

    # data
    train_dataset = LPRDataLoader(args.train_img_dirs, args.img_size, args.lpr_max_len, augment=True)
    test_dataset = LPRDataLoader(args.test_img_dirs, args.img_size, args.lpr_max_len, augment=False)
    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # opt,lr,loss
    optimizer = Lars(noBiasDecay(lprnet, lr=args.learning_rate, weight_decay=args.weight_decay), momentum=args.momentum)
    # optimizer = torch.optim.RMSprop(noBiasDecay(lprnet, lr=args.learning_rate, weight_decay=args.weight_decay))
    scheduler = CosineDecayLR(optimizer, T_max=max_iter, lr_init=args.learning_rate, lr_min=0.00001, warmup=0)
    # ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    focal_ctc_loss=FocalCTCLoss(blank=len(CHARS)-1, reduction='mean',gamma=2)

    best_acc=0.0
    infor= {'best_iteration':0,'last_iteration':0,'best_acc':0.0} ##
    loss_500_batchs = 0.0
    print(f'traindata {args.train_img_dirs} testdata: {args.test_img_dirs}')
    print(f'trainset:{len(train_dataset)} testset:{len(test_dataset)} batch_size:{args.train_batch_size} epoch:{args.max_epoch}')
    print(f'iters:{max_iter} save_interval:{args.save_interval} test_interval:{args.test_interval} ')

    # training
    lprnet.train()
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_train))
            loss_val = 0
            epoch += 1

        # save last.pt
        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'last.pth')
            infor['last_iteration']=iteration
            print(f'last.pt saved to {args.save_folder}')

        # test and save best.pt
        if (iteration + 1) % args.test_interval == 0:
            lprnet.eval()
            test_acc=Greedy_Decode_Eval(lprnet, test_dataset, args)
            if best_acc<test_acc and test_acc>0.8:
                best_acc = test_acc
                torch.save(lprnet.state_dict(),args.save_folder+f'best_{round(best_acc,4)}.pth')
                infor['best_iteration'] = iteration
                infor['best_acc'] =best_acc
                print(f'best.pt saved to {args.save_folder}')
            lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load batch
        images, top_labels, bot_labels,top_lengths,bot_lengths = next(batch_iterator)

        # update lr
        scheduler.step(iteration)
        lr = optimizer.param_groups[0]['lr']

        if args.cuda:
            images = Variable(images, requires_grad=False).to(device)
            top_labels = Variable(top_labels, requires_grad=False).to(device)
            bot_labels = Variable(bot_labels, requires_grad=False).to(device)
        else:
            images = Variable(images, requires_grad=False)
            top_labels = Variable(top_labels, requires_grad=False)
            bot_labels = Variable(bot_labels, requires_grad=False)

        # forward
        logits = lprnet(images) #
        top=logits[:,:,0,:]
        bot=logits[:,:,1,:]
        top = top.permute(2, 0, 1).log_softmax(2) # for ctc loss: T x B x C  # requires_grad_()
        bot = bot.permute(2, 0, 1).log_softmax(2)

        # backprop
        optimizer.zero_grad()
        top_loss = focal_ctc_loss(top, top_labels, input_lengths=(18,)*args.train_batch_size, target_lengths=(2,)*args.train_batch_size)  # focal_CTC_LOSS
        bot_loss = focal_ctc_loss(bot, bot_labels, input_lengths=(18,)*args.train_batch_size, target_lengths=(5,)*args.train_batch_size)  # focal_CTC_LOSS
        loss = top_loss + bot_loss
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        # log
        loss_500_batchs+=loss.item()
        if iteration % 500 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss_500_batchs/500) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
            loss_500_batchs=0.0
    # # final test
    # print("Final test Accuracy:")
    # Greedy_Decode_Eval(lprnet, test_dataset, args)
    #
    # # save final parameters
    # torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    print(f'model saved to {args.save_folder}')
    print(infor)

def collate_fn_train(batch):
    '''
    dataloader 后处理
    :param batch:
    :return:
    '''
    imgs = []
    top_labels = []
    bot_labels = []
    top_lengths = []
    bot_lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample # length=7
        imgs.append(torch.from_numpy(img))
        top_labels.extend(label[:2])
        bot_labels.extend(label[2:])
        top_lengths.append(2)
        bot_lengths.append(5) # (5,)*args.train_batch_size
    # labels = np.asarray(labels).flatten().astype(np.int32)
    top_labels = torch.tensor(top_labels) # 1*(2*B)
    bot_labels = torch.tensor(bot_labels) # 1*(5*B)

    return (torch.stack(imgs, 0), top_labels, bot_labels,top_lengths,bot_lengths) # 128*3*24*94


def collate_fn_test(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths) # 128*3*24*94 , 1*(7*16), 1*16

def sparse_tuple_for_ctc(T_length, lengths):
    '''

    :param T_length: 18
    :param lengths: [7]*16
    :return:
    '''
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_test))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()

    # per batch
    for i in range(epoch_size):
        # print(f'{i}/{epoch_size}:')
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images) #

        # greedy decode
        prebs = prebs.permute(2,0,1,3).cpu().detach().numpy() # to 2*B*71*18
        top = prebs[0,...]
        bot = prebs[1,...]
        prebs = np.concatenate((top,bot),axis=2) # to B*71*36

        preb_labels = list()
        # per img in a batch
        for i in range(prebs.shape[0]):
            # to preb_label 1*36
            preb = prebs[i, :, :] # 71*36
            preb_label = list()
            for j in range(preb.shape[1]):  # C
                preb_label.append(np.argmax(preb[:, j], axis=0))
            # to no_repeat_blank_label
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeated label and blank label !
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)


        # count
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

    return Acc #


if __name__ == "__main__":
    with open('train_LPRNet_dp.py') as f:
        print(f.read())

    sta_date=time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    print(f'started at {sta_date}')
    train()

    fin_date = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    print(f'finished at {fin_date}')