import torch.nn as nn
import torch
from model import SixDRepNet
#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)

class Distilling_loss1(nn.Module):
    '''
    蒸馏学习 loss1
    '''
    def __init__(self):
        super(Distilling_loss1, self).__init__()
        self.teacher = SixDRepNet(backbone_name='RepVGG-B1g2',
                           backbone_file='',
                           deploy=True,
                           pretrained=False,teacher=True)
        self.teacher.load_state_dict(torch.load('/home/xiancai/face_angle/6DRepNet/results/other/6DRepNet_300W_LP_AFLW2000.pth'))
        self.teacher.cuda(0)
        self.teacher.eval()
        self.mse = nn.MSELoss()
        self.tea_feats={}

    def forward(self,img,stu_feat):
        '''

        :param img: 默认h*w: 224*224
        :param stu_feat: 1*2048
        :return:
        '''
        tea_feat = self.teacher(img) #
        loss1 = self.mse(stu_feat,tea_feat)
        return loss1
