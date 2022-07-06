import os

import torch
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR

import time
import numpy as np

from models.resnet import *
from dataset import Dataset
from config.config import Config
from loss.focal_loss import FocalLoss
from util.lr import CosineDecayLR # lr余弦衰减

from myutil import util_time  # 记录时间
import json

def eval_train(model, criterion, testloader):
	model.eval()
	test_loss = 0.0 # cost function error
	correct = 0.0

	for (datas, labels) in testloader:
		# if not datas.shape[0]==opt.test_batch_size:
		# 	print(f'test batch size :{datas.shape[0]}')
		# 	raise
		datas = datas.to(device)
		labels = labels.to(device).long()
		outputs = model(datas)
		loss = criterion(outputs, labels)
		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()

	acc=correct.float() / len(test_dataset)
	loss=test_loss / len(testloader)
	print('Test set: loss_per_img: {:.4f}, acc_val: {:.4f}'.format(loss,acc))
	model.train()
	return acc

if __name__ == '__main__':

	ut=util_time()
	ut.info_start_time()

	# init
	opt = Config()

	# load data
	test_dataset = Dataset(opt.test_root, opt.test_list, phase='test', input_shape=opt.input_shape)
	testloader = data.DataLoader(test_dataset,batch_size=opt.test_batch_size,
								 shuffle=False,pin_memory=True,num_workers=opt.num_workers)
	
	train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
	trainloader = data.DataLoader(train_dataset,
	                              batch_size=opt.train_batch_size,
	                              shuffle=True,
	                              pin_memory=True,
	                              num_workers=opt.num_workers)
	# loss
	if opt.loss == 'focal_loss':
		criterion = FocalLoss(gamma=2)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# model
	if opt.model_name=='resnet18':
		model = resnet18()
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 0号gpu不用
	if torch.cuda.device_count() > 1:
		model=nn.DataParallel(model) # 使用多gpu训练模型
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	if opt.finetune == True:
		pretrained_dict = torch.load(opt.load_model_path)
		# 兼容单卡多卡训练（多卡训练的模型有'module.'前缀，单卡的没有）
		if torch.cuda.device_count() > 1:
			dataparaller_pretrained_dict={}
			for k,v in pretrained_dict.items():
				if k.split('.')[0]=='module':
					dataparaller_pretrained_dict[k]=v
				else:
					dataparaller_pretrained_dict['module.'+k]=v
			pretrained_dict=dataparaller_pretrained_dict
		model.load_state_dict(pretrained_dict)
		print('load weights from pretrained model')

	# optimizer, lr
	if opt.optimizer == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
	scheduler = MultiStepLR(optimizer, milestones = opt.milestones, gamma=0.1)
	# scheduler = CosineDecayLR(optimizer, T_max=len(trainloader) * opt.max_epoch, lr_init=opt.lr, lr_min=1e-5,
							  # warmup=opt.warmup * len(trainloader))

	# train, test, save
	print(opt.__dict__)
	print(f'trainset_path:{opt.train_list}\ntestset_path:{opt.test_list}')
	print(f'trainset:{len(train_dataset)}\ntestset:{len(test_dataset)}')
	print('{} train iters per epoch in dataset'.format(len(trainloader)))
	best_acc=0
	info_loss = 0
	start = time.time()
	for epoch in range(0, opt.max_epoch):
		#
		# train(model, criterion, optimizer, scheduler, trainloader, epoch)
		# # save model
		# if epoch % opt.save_interval == 0 or epoch == (opt.max_epoch - 1):
		# 	torch.save(model.state_dict(), 'checkpoints/model-epoch-'+str(epoch) + '.pth')
		# 	eval_train(model, criterion, testloader)

		model.train()
		for ii, data in enumerate(trainloader):
			# train: forward and backward, update grad and lr
			# print(ii)
			data_input, label = data
			data_input = data_input.to(device)
			label = label.to(device).long()
			output = model(data_input)
			loss = criterion(output, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			iters = epoch * len(trainloader) + ii+1
			# scheduler.step(iters+1) # lr
			scheduler.step(epoch)  # lr

			info_loss+=loss.item()
			if iters % opt.print_freq == 0:
				# train_loss train_acc
				output = output.data.cpu().numpy()
				output = np.argmax(output, axis=1)
				label = label.data.cpu().numpy()
				acc = np.mean((output == label).astype(int))
				total_time = (time.time() - start)
				time_str = time.asctime(time.localtime(time.time()))
				le_r=optimizer.param_groups[0]['lr']
				print(f'{time_str}: epoch: {epoch}, iters: {iters},total_time: {round(total_time/3600,4)}h, lr: {le_r},loss_per_img: {info_loss/opt.print_freq}, acc_one_batch: {acc}')
				info_loss=0

		# test
		test_acc=eval_train(model, criterion, testloader)
		# save
		torch.save(model.state_dict(), f'{opt.save_path}last.pth')
		print(f'last model saved to {opt.save_path}last.pth')
		if test_acc>0.90 and test_acc>best_acc:
			best_acc=test_acc
			best_save_path=f'{opt.save_path}{opt.model_name}_best_{round(best_acc.item(), 4)}_epoch{epoch}.pth'
			torch.save(model.state_dict(),best_save_path )
			print(f'Best model saved to {best_save_path}')

	ut.info_finish_time()
	ut.info_total_time()
