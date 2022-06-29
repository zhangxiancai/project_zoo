import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,5) #输入3*32*32
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,10) #accuracy:0.6555
        # self.fc2=nn.Linear(100,10)
        # nn.BatchNorm1d()

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        # x=x.view(-1,x.numel())
        x=x.view(-1,16*5*5)
        x=self.fc1(x) #accuracy:0.66

        return x

# #定义模型
# net=Net()
# print(net)
#
# #forward
# img=torch.randn(1,1,32,32)
# out=net(img)
#
# #backward  损失函数+标签+out==>>梯度
# mse=nn.MSELoss()
# tar=torch.randn(1,10)#标签
# loss=mse(out,tar)
# loss.backward()
#
# #更新权重
# opt=optim.SGD(net.parameters(),lr=0.01)
# print(net.conv1.bias)
# print(net.conv1.bias.grad)
# opt.step()
# print(net.conv1.bias)
# print(net.conv1.bias.grad)

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# torch.multiprocessing.set_sharing_strategy('file_system') #testloader

#设备
device=torch.device('cuda:0')
total_device=torch.cuda.device_count()
print(f'total of devices:{total_device}')

#数据集载入和预处理
batch_size=32
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((112,112)),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=torchvision.datasets.CIFAR10(root='./',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='./',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=0)
ns=len(trainloader)
print(f'train images:{len(trainset)},test images:{len(testset)},batchsize:{batch_size},train_batchs:{ns}')

#模型
# pre_model='net-best-0.6556.pt'
# net=Net()
# if pre_model:
#     net.load_state_dict(torch.load(pre_model))
#     print(f'using pretrained model:{pre_model}')

import resnet
net=resnet.resnet18()
net.fc=nn.Linear(256,10)
# net.load_state_dict(torch.load('./net-best-0.6488.pt'))
# print('net.load_state_dict')
if total_device>1:
    net=nn.DataParallel(net)#
net.to(device) #

#优化算法和损失函数
op=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
p=nn.CrossEntropyLoss()
sched=optim.lr_scheduler.ExponentialLR(op,gamma=0.9)

#训练
t0 = time.time() #
for epoch in range(20):
    # t1=time.time()
    net.train()
    running_loss=0.0
    for i, data in enumerate(trainloader):
        batch_imgs,batch_labs=data
        batch_imgs=batch_imgs.to(device)#gpu
        batch_labs=batch_labs.to(device)#
        op.zero_grad()
        batch_outs=net(batch_imgs)
        loss=p(batch_outs,batch_labs) #
        # print(net.conv1.bias)
        # print(net.conv1.bias.grad)
        loss.backward()
        # print(net.conv1.bias.grad)
        op.step()
        # print(net.conv1.bias)
        running_loss+=loss.item()
        # if i%2000==1999:
        if True:
            print(f'epoch:{epoch},batch:{i},lr:{sched.get_last_lr()}loss:{running_loss/2000}')
            running_loss=0
    net.eval()
    acc = 0.0
    total = 0
    max_acc=0.0 #
    test_loss = 0.0
    for imgs, labs in testloader:
        outs = net(imgs)
        _, pre = torch.max(outs, 1)  #

        total += outs.size(0)
        # print(f'pre:{pre.device},labs:{labs.device}')  pre.to('cpu')  ##
        # print(f'pre:{pre.device}')
        temp = (labs == pre)
        acc += temp.sum().item()
        test_loss += p(outs, labs)
    acc = acc / total
    print(f'test_loss:{test_loss/len(testloader)} accuracy:{acc},time:{time.time()-t0}s')
    if max_acc<acc:
        best_dict=net.state_dict()
        max_acc=acc
    sched.step()# 调整lr

# 保存
# torch.save(best_dict, f'net-best-{max_acc}.pt')
# print(f'best model saved to net-best-{max_acc}.pt')

torch.save(best_dict, f'resnet18-best-{max_acc}.pt')
print(f'best model saved to resnet18-best-{max_acc}.pt')

print('Done.')



# #统计在测试集上的精度
# acc=0.0
# total=0
# res=torch.zeros((10,10),dtype=int)
# # with torch.no_grad():
# for imgs, labs in testloader:
#     outs=net(imgs)
#     _,pre=torch.max(outs,1) #
#
#     res[labs,pre]+=1#
#     total+=outs.size(0)
#     # print(f'pre:{pre.device},labs:{labs.device}')
#     pre=pre.to('cpu')##
#     # print(f'pre:{pre.device}')
#     temp=(labs==pre)
#     acc+=temp.sum().item()
# print(f'tatal of test images:{total} accuracy:{acc/total}')
# print(res.numpy())
# for i in range(10):
#     print(f'acc of class {i}: {res[i,i]/(res[i,...].sum())}')



