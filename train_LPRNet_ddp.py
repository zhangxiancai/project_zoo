# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
单机多gpu：DistributedDataParallel
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 train_LPRNet_ddp.py
'''
from tqdm import tqdm

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import LPRNet
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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from contextlib import contextmanager
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    让指定代码段先被processe0执行一遍，随后才允许其余processe进入
    """
    if local_rank !=0: # -1: dp,  0,1,2...: ddp
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0]) #等待所有进程到达栅栏处，才释放


def get_parser():

    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=100, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs',default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/train.txt",
                                                    "/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/jv_clean_plate_train.txt",
                                                    '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/yellow_plate_train.txt'],help='the train images path')
    parser.add_argument('--test_img_dirs', default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt",
                                                    "/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/jv_clean_plate_val.txt",
                                                    '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/yellow_plate_val.txt'],help='the test images path')
    # parser.add_argument('--train_img_dirs', default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/yellow_plate_train.txt"], help='the train images path')
    # parser.add_argument('--test_img_dirs', default=["/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/yellow_plate_val.txt"], help='the test images path')

    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.01, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, help='training batch size.')
    parser.add_argument('--test_batch_size', default=32, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    # parser.add_argument('--save_interval', default=2, type=int, help='interval for save model state dict')
    # parser.add_argument('--test_interval', default=2, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model', default='/home/xiancai/plate/LPRNet_Pytorch/Result/2022_04_20_more_agu_data/best_0.9828.pth', help='pretrained base model')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int) # for ddp
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths) # 128*3*24*94 , ~*1, 128*1

def collate_fn_debug(batch):
    imgs = []
    labels = []
    lengths = []
    filenames=[]
    for _, sample in enumerate(batch):
        img, label, length, filename = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        filenames.append(filename)
    labels = np.asarray(labels).flatten().astype(np.int32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths,filenames) # 128*3*24*94 , ~*1, 128*1

def train(args):

    # init
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    if not torch.cuda.device_count() > 1:
        raise Exception(f'DDP needs multi gpus, but now : {torch.cuda.device_count()}')
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    if args.local_rank == 0:
        args.save_folder=f'weights/{str(int(time.time()))}/' # init the dir to save model
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

    T_length = 18 # args.lpr_max_len

    # model
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    # print("MODEL: Successful to build network!")

    if args.pretrained_model: # initial weights
        #lprnet.load_state_dict(torch.load(args.pretrained_model))
        load_model(lprnet, args.pretrained_model)
        print("load pretrained model successful!")
    # else:
    #     def xavier(param):
    #         nn.init.xavier_uniform(param)
    #
    #     def weights_init(m):
    #         for key in m.state_dict():
    #             if key.split('.')[-1] == 'weight':
    #                 if 'conv' in key:
    #                     nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
    #                 if 'bn' in key:
    #                     m.state_dict()[key][...] = xavier(1)
    #             elif key.split('.')[-1] == 'bias':
    #                 m.state_dict()[key][...] = 0.01
    #
    #     lprnet.backbone.apply(weights_init)
    #     lprnet.container.apply(weights_init)
    #     print("initial net weights successful!")

    lprnet = nn.parallel.DistributedDataParallel(lprnet, device_ids=[args.local_rank],output_device=args.local_rank) # for ddp
    if args.sync_bn: # SyncBatchNorm
        lprnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lprnet).to(device)
        print('Using SyncBatchNorm()')

    # data
    with torch_distributed_zero_first(args.local_rank):  # init dataset *.cache only once if DDP
        train_dataset = LPRDataLoader(args.train_img_dirs, args.img_size, args.lpr_max_len, augment=True)
    train_sampler = DistributedSampler(train_dataset) # for ddp
    train_dataloader = (DataLoader(train_dataset, args.train_batch_size//torch.distributed.get_world_size(), shuffle=False, sampler=train_sampler,
                                 num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True))
    if args.local_rank==0:
        test_dataset = LPRDataLoader(args.test_img_dirs, args.img_size, args.lpr_max_len, augment=False)
    epoch_size = len(train_dataset) // args.train_batch_size # 一个epoch的batch数
    max_iter = args.max_epoch * epoch_size

    # opt,lr,loss
    optimizer = Lars(noBiasDecay(lprnet, lr=args.learning_rate, weight_decay=args.weight_decay), momentum=args.momentum)
    # optimizer = torch.optim.RMSprop(noBiasDecay(lprnet, lr=args.learning_rate, weight_decay=args.weight_decay))
    scheduler = CosineDecayLR(optimizer, T_max=max_iter, lr_init=args.learning_rate, lr_min=0.00001, warmup=0)
    # ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    focal_ctc_loss=FocalCTCLoss(blank=len(CHARS)-1, reduction='mean',gamma=2)

    best_acc=0.0
    infor= {'best_iteration':0,'last_iteration':0,'best_acc':0.0} ##
    if args.local_rank==0:
        print(f'train_data {args.train_img_dirs}\ntotal:{len(train_dataset)}\ntrain_batch_size:{args.train_batch_size}')
        print(f'test_data:{args.test_img_dirs}\ntotal:{len(test_dataset)}\ntest_batch_size:{args.test_batch_size} ')
        # print(f'save_interval:{args.save_interval} test_interval:{args.test_interval} ')
        print(f'epochs:{args.max_epoch}')

    # training
    lprnet.train()
    # per epoch
    for epoch in range(args.max_epoch):
        train_sampler.set_epoch(epoch) # 重置sampler
        if args.local_rank == 0:
            task=tqdm(enumerate(train_dataloader), total=len(train_dataloader),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            task=enumerate(train_dataloader)
        # per batch
        for ind,(images, labels, lengths) in task:
            # print(images.shape) # 16*3*24*94
            images = images.to(device)
            labels = labels.to(device)

            # forward
            logits = lprnet(images)  # 注意
            log_probs = logits.permute(2, 0, 1).log_softmax(2)  # for ctc loss: T x N x C
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)  # 注意
            # loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths) #  CTC_LOSS
            loss = focal_ctc_loss(log_probs, labels, input_lengths=input_lengths,
                                  target_lengths=target_lengths)  # focal_CTC_LOSS

            # backward
            optimizer.zero_grad()
            loss.backward() # 在这里同步多个GPU上的梯度

            # update
            iteration=epoch * epoch_size + ind
            scheduler.step(iteration)
            optimizer.step() # 不在这里同步梯度

            # log
            if args.local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                s='Epoch:' + f'{epoch}/{args.max_epoch-1}' + '|| Current iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item())  + 'LR: %.8f' % (lr)
                task.set_description(s)
            # print(f'rank{args.local_rank} epoch{epoch} log')
        # save last.pt
        if args.local_rank == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'last.pth')
            infor['last_iteration'] = iteration

        # test and save best.pt
        if args.local_rank == 0 and epoch > 5:
            lprnet.eval()
            test_acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
            if best_acc < test_acc and test_acc > 0.8:
                best_acc = test_acc
                torch.save(lprnet.state_dict(), args.save_folder + f'best_{round(best_acc, 4)}.pth')
                infor['best_iteration'] = iteration
                infor['best_acc'] = best_acc
            lprnet.train()  # should be switch to train mode

    print(f'model saved to {args.save_folder}')
    print(infor)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

@torch.no_grad()
def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    # ma = 0 # debug
    # debug_res_pt=[]
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

        # DEBUG = True
        # if DEBUG: # check input
        #     import classify_plate
        #     img_onnx=classify_plate.classify_plate('/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/202101020-ok/752cfe281e3a79afdbaab20bb9abdbf8_苏E637AF.jpg')
        #     print(f'img: {abs(img_onnx-images.numpy()).max()}')
        # forward
        prebs = Net(images) #

        # debug check model
        # DEBUG=True
        # if DEBUG:
        #     import classify_plate
        #     pre_onnx = classify_plate.ses.run(None, {'data': images.numpy()})[0]
        #     # print(pre.shape)
        #     pre_onnx = np.mean(pre_onnx, axis=2)
        #     pre_pt=prebs.cpu().detach().numpy()
        #     print(pre_pt.shape)
        #     print(pre_onnx.shape)
        #     ma=max(abs(pre_pt-pre_onnx).max(),ma)
        #     print(ma)

        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            # to preb_label
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):  # C
                preb_label.append(np.argmax(preb[:, j], axis=0))
            # to no_repeat_blank_label
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label 注意
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        # if DEBUG:
        #     debug_res_pt.extend(preb_labels)

        # count
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1 # 长度不等
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1 # 长度相等 字符不一致
    # if DEBUG:
    #     return debug_res_pt

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("\t[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)),end='')
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

    return Acc #
@torch.no_grad()
def Greedy_Decode_Eval_debug(Net, datasets, args):

    DEBUG = False
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn_debug))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    tag=0
    err_name=[]
    t1 = time.time()
    ma = 0 # debug
    strs=[]
    debug_res_pt=[]
    for i in range(epoch_size):
        print(f'{i}/{epoch_size}:')
        # load train data
        images, labels, lengths,filename = next(batch_iterator) #
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
        img_or = torch.tensor(images)
        print(f'img: {abs(img_or.numpy() - images.numpy()).max()}')
        DEBUG = True
        if DEBUG: # check input
            import classify_plate
            img_onnx=classify_plate.preproccess(filename[0])
            print(f'img: {abs(img_onnx-images.numpy()).max()}')
        # forward
        prebs = Net(images) #

        # # debug check model

        # if DEBUG:
        #     import classify_plate
        #     pre_onnx = classify_plate.ses.run(None, {'data': images.numpy()})[0]
        #     # print(pre.shape)
        #     pre_onnx = np.mean(pre_onnx, axis=2)
        #     pre_pt=prebs.cpu().detach().numpy()
        #     print(pre_pt.shape)
        #     print(pre_onnx.shape)
        #     ma=max(abs(pre_pt-pre_onnx).max(),ma)
        #     print(ma)

        # greedy decode
        prebs = prebs.cpu().detach().numpy()

        preb_labels = list()
        for i in range(prebs.shape[0]):
            print(f'prebs_shape:{prebs.shape[0]} i:{i}')
            if filename[i].split('/')[-1] == '0066-1_1-302&495_428&539-428&536_304&539_302&498_426&495-0_0_33_6_26_24_30-69-2-皖A9G206.jpg':
                print(prebs.shape)
                print(prebs[:,0,:])
                images=images.numpy()
                print(images.shape)
                print(images[0,0,:,1])
                te=0
            # to preb_label
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):  # C
                preb_label.append(np.argmax(preb[:, j], axis=0))
            # to no_repeat_blank_label
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label 注意
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        if DEBUG:
            debug_res_pt.extend(preb_labels)

        # count
        for i, label in enumerate(preb_labels):
            # if len(label) != len(targets[i]):
            #     Tn_1 += 1
            #     continue
            # if (np.asarray(targets[i]) == np.asarray(label)).all():
            #     Tp += 1
            # else:
            #     Tn_2 += 1
            print(f'preb_labels:{len(preb_labels)} i:{i}')
            print(np.asarray(targets[i]))
            print(np.asarray(label))
            tru_lab = ''.join([CHARS[ss] for ss in targets[i]])
            pre_lab = ''.join([CHARS[ss] for ss in label])
            # print(filename)
            nam=filename[i][:-4]+'-pre'+pre_lab+'-tar'+tru_lab+'.jpg'
            print(nam)
            strs.append(nam)
            if len(label) != len(targets[i]) or not (np.asarray(targets[i]) == np.asarray(label)).all():
                err_name.append(nam)
                tag+=1
    # print(strs)
    # print(err_name)
    # save strs
    with open('/data1/xiancai/PLATE_DATA/res_classify_err/debug_pt_res.txt','w')  as f:
        f.write('\n'.join(strs))
    with open('/data1/xiancai/PLATE_DATA/res_classify_err/debug_pt_res_err.txt','w')  as f:
        f.write('\n'.join(err_name))
    if DEBUG:
        # return debug_res_pt
        print(err_name)
        return tag
    # Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    # print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    # t2 = time.time()
    # print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    print(f'tag:{tag}')
    # return Acc #


if __name__ == "__main__":
    args = get_parser()
    if args.local_rank == 0:
        sta_date=time.strftime("%Y-%m-%d %H:%M", time.localtime())
        print(f'started at {sta_date}')
    train(args)
    #
    # world_size = 4
    # mp.spawn(train,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True)
    if args.local_rank == 0:
        fin_date = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        print(f'finished at {fin_date}')