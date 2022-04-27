import numpy as np
import torch
#from torch.utils.serialization import load_lua
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y


    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector( v, use_gpu=True):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))  
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw, use_gpu) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z, use_gpu)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices, use_gpu=True):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
    if use_gpu:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
    else:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3))  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R



import torch.nn as nn
def noBiasDecay(model, lr, weight_decay):
    '''
    no bias decay : only apply weight decay to the weights in convolution and fully-connected layers
    In paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
    Ref: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/blob/master/utils.py
    '''
    decay, bias_no_decay, weight_no_decay = [], [], []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                bias_no_decay.append(m.bias)
        else:
            if hasattr(m, 'weight'):
                weight_no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                bias_no_decay.append(m.bias)

    assert len(list(model.parameters())) == len(decay) + len(bias_no_decay) + len(weight_no_decay)

    # bias using 2*lr
    return [{'params': bias_no_decay, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': weight_no_decay, 'lr': lr, 'weight_decay': 0.0},
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]


def convert_half():
    '''
    制作预训练模型A0-half
    :return:
    '''
    checkpoint = torch.load('/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0-train.pth')
    checkpoint_half={}
    flag_pre = None
    for k,v in checkpoint.items():
        s=  2/3 # 比例
        # convert
        print(v.shape)

        if len(v.shape) == 0: # bn.num_batches_tracked
            checkpoint_half[k]=v
        elif len(v.shape)==1: # bn.weight bn.bias bn.running_mean bn.running_var
            flag = int(v.shape[0] * s)
            checkpoint_half[k] = torch.tensor(v[:flag])
        else: # conv.weight linear.weight
            # flag_pre=int(v.shape[0] * s) if not flag_pre else flag_pre # 上一层c
            # flag0 = int(v.shape[0] * s)
            # flag1 = flag_pre
            # flag_pre=flag0
            flag0 = int(v.shape[0] * s)
            flag1 = int(v.shape[1] * s) #
            if v.shape[1]==3: # 第一层conv.weight的c=3保持不变
                checkpoint_half[k] = v[:flag0,...].clone()
            else:
                checkpoint_half[k] = v[:flag0,:flag1,...].clone()

    # save
    torch.save(checkpoint_half,'/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0-Half-train.pth')


def convert_A0s():
    '''
    制作预训练模型A0s
    :return:
    '''
    checkpoint = torch.load('/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0-train.pth')
    checkpoint_A0s={}
    layers=['stage3.6','stage3.7','stage3.8','stage3.9','stage3.10','stage3.11','stage3.12','stage3.13'] # 删除的层
    for k,v in checkpoint.items():
        if not (k[:8] in layers or k[:9] in layers ):
            checkpoint_A0s[k]=v.clone()

    # save
    torch.save(checkpoint_A0s,'/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0s-train.pth')

def convert_A0s_stu():
    '''
    A0s_stu预训练模型
    :return:
    '''
    checkpoint = torch.load('/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0-train.pth')
    checkpoint_A0s={}
    layers=['stage3.6','stage3.7','stage3.8','stage3.9','stage3.10','stage3.11','stage3.12','stage3.13','stage4.0','linear.w','linear.b'] # 删除的层
    for k,v in checkpoint.items():
        if not (k[:8] in layers or k[:9] in layers ):
            checkpoint_A0s[k]=v.clone()

    # save
    torch.save(checkpoint_A0s,'/home/xiancai/face_angle/6DRepNet_04/models/RepVGG-A0s-stu-train.pth')

def compare_weights():
    '''
    比较两个相同结构模型的权重差异
    :return:
    '''
    pth1=torch.load('/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649297858_bs64/RepVGG-A0_epoch_101_mae13.2071.pth')
    pth2=torch.load('/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649297858_bs64/RepVGG-A0_epoch_177_mae13.5007.pth')

    for k,v in pth1.items():
        d=torch.mean(torch.abs(pth2[k]-pth1[k]))
        print(k,d)

if __name__=='__main__':
    compare_weights()