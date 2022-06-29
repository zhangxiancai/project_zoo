'''
神经风格转换
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim

from resnet import resnet18

# import PIL.Image as Image
import cv2


class ContentandStyleLoss(nn.Module):
    def __init__(self,c_img,s_img):
        super(ContentandStyleLoss, self).__init__()
        self.c_img=c_img
        self.s_img=s_img
        self.g_img=nn.Parameter(torch.rand([1,3,112,112]))
        # self.g_img=nn.Parameter(self.img_load(self.c_img))
        self.cnn = resnet18(num_classes=114)

        model_addr='/home/xiancai/fire-equipment-demo/firecontrol_classification/result/2021_12_06/fire_classify_resnet18_cls114_12_06.pth'
        st_dict = torch.load(model_addr)
        self.cnn.load_state_dict(remove_prefix(st_dict, 'module.'))
        self.cnn.eval()

    def forward(self):
        # 损失函数
        ls=['conv1','conv2_x','conv3_x','conv4_x','conv5_x']
        loss,c_loss,s_loss=0,0,0
        for i in ls:
            cnn_c_img=self.cnn(self.img_load(self.c_img), out_layer=i).detach_()
            cnn_s_img = self.cnn(self.img_load(self.s_img), out_layer=i).detach_()
            cnn_g_img=self.cnn(self.g_img, out_layer=i)
            c_loss+=F.mse_loss(cnn_c_img,cnn_g_img)
            s_loss+=F.mse_loss(self.gram_matrix(cnn_s_img),self.gram_matrix(cnn_g_img))

        loss+=0.5*c_loss+0.5*s_loss
        # print(cnn_c_img.max())
        # print(cnn_c_img.min())
        # print(cnn_g_img.max())
        # print(cnn_g_img.min())
        # print(self.gram_matrix(cnn_s_img).max())
        # print(self.gram_matrix(cnn_s_img).min())
        # print(self.gram_matrix(cnn_g_img).max())
        # print(self.gram_matrix(cnn_g_img).min())
        return loss, c_loss, s_loss # 返回loss

    def gram_matrix(self,input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # 特征映射 b=number
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # 我们通过除以每个特征映射中的元素数来“标准化”gram矩阵的值.
        G=G.div(c*d)
        # G = G.div(a * b * c * d)
        return G

    def img_load(self, img):
        # img=Image.open(img_addr)
        img=pre_process(img)
        img.permute(2,0,1)
        img=img.view(1,3,112,112) #
        return img

pre_process=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    #transforms.RandomCrop((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

])

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


if __name__=='__main__':
    # init
    c_img=cv2.imread('2_lab8.jpg')
    s_img = cv2.imread('微信截图_20211207160955.png')
    loss_module=ContentandStyleLoss(c_img,s_img)

    #优化算法
    lr=4
    # op = optim.SGD(loss_module.g_img, lr=1000,momentum=0.9)
    # sched = optim.lr_scheduler.ExponentialLR(op, gamma=0.9)
    # 迭代
    for ep in range(5000):
        c_s_loss,c_loss,s_loss=loss_module()
        c_s_loss.backward()
        print(f'ep:{ep},loss:{c_s_loss},c_loss:{c_loss},s_loss:{s_loss},lr:{lr}')
        # step
        with torch.no_grad():
            if ep%300==0:
                lr*=0.95**(ep//300)
            loss_module.g_img-=lr*loss_module.g_img.grad
            loss_module.g_img.grad.zero_()

    # save
    res_img=loss_module.g_img.detach().numpy()
    res_img=cv2.convertScaleAbs(res_img*127.5+127.5).transpose(0,2,3,1).squeeze(axis=0)
    out='gene_img.jpg'
    flag=cv2.imwrite(out,res_img)
    print(flag)
    print(f'gene img saved to {out}')
    print('Done.')