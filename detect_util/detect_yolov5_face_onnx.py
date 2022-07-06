'''
人脸检测, 检测一个文件夹的图片, 使用onnx模型
'''
# import sys
# sys.path.append('./')
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

sys.path.append('/home/xiancai/face_detection/yolov5-face/')
# sys.path.insert(0,'/home/xiancai/face_detection/yolov5-face/')
# print(sys.path)
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import onnxruntime as ort

# sys.path.remove('/home/xiancai/face_detection/yolov5-face/')


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding 最少填充
    # elif scaleFill:  # stretch 长宽比变化
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def detect_one_onnx(orgimg):

    # load img
    img0 = copy.deepcopy(orgimg)

    # resize origin image
    h0, w0 = orgimg.shape[:2]  # orig hw
    imgsz=img_size
    img = letterbox(img0, new_shape=imgsz)[0]
    # cv2.imwrite('/home/xiancai/DATA/PLATE_DATA/1.jpg',img)
    # BGR to RGB ,to 1*3*h*w
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = img[None,...] # to 1*3*~*~
    img =  np.ascontiguousarray(img).astype(np.float32)  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0


    # Inference
    t0 = time.time()
    outs=ses.run(['strid_8','strid_16','strid_32'],{'input':img})
    t1 = time_synchronized()
    # print(t1-t0)

    # onnx 输出转换
    anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433] ]
    grid = [torch.zeros(1)] * 3  # 3 layer
    stride = torch.tensor([8.0, 16.0, 32.0])
    a = torch.tensor(anchors).float().view(3, -1, 2)  # 3 layer
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)  # 3 layer
    cls=1
    z = []
    for i in range(3):
        # reshape
        x = torch.tensor(outs[i]) # 1*c*h*w
        x = x.view(1,3,-1,x.shape[2],x.shape[3]).permute(0,  1, 3, 4, 2).contiguous()  # 1*3*~*~*17
        # convert
        ny, nx = x.shape[2], x.shape[3]
        if grid[i].shape[2:4] != x.shape[2:4]:
            grid[i] = make_grid(nx=nx, ny=ny).to(x.device)
        y=x
        y[..., 0:2] = (y[..., 0:2].sigmoid() * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4].sigmoid() * 2) ** 2 * anchor_grid[i]  # wh
        y[..., 4] = y[..., 4].sigmoid() # conf
        y[..., 5:7] = y[..., 5:7] * anchor_grid[i] + grid[i] * stride[i]  # landmark x1 y1
        y[..., 7:9] = y[..., 7:9] * anchor_grid[i] + grid[i] * stride[i]  # landmark x2 y2
        y[..., 9:11] = y[..., 9:11] * anchor_grid[i] + grid[i] * stride[i]  # landmark x3 y3
        y[..., 11:13] = y[..., 11:13] * anchor_grid[i] + grid[i] * stride[i]  # landmark x4 y4
        y[..., 13:15] = y[..., 13:15] * anchor_grid[i] + grid[i] * stride[i]  # landmark x5 y5
        y[..., 15] = y[..., 15].sigmoid() # cls
        z.append(y.view(1, -1, cls + 5+10))  # 17
    z = torch.cat(z, 1)  # 1*~*17

    # Apply NMS
    pred = non_max_suppression_face(z, conf_thres, iou_thres)[0] # [0]=img0
    det = pred# ~*16 (x1, y1, x2, y2, conf, marks1-10,cls)

    # Process detections
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round() # x1y1x2y2 to 原图像素
        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round() # landmarks to 原图像素
    return det

def show_results(img, xywh, conf, landmarks, class_num):
    # 画box和landmarks
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5): # 画五对marks
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), i*2+2, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_box_landmarks(orgimg,det):
    # 画图 box,lands
    gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
    for j in range(det.shape[0]):
        xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()  # to xywh 原图ratio
        conf = det[j, 4].cpu().numpy()
        landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()  # to 原图ratio
        class_num = det[j, 15].cpu().numpy()
        orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
    return orgimg

def draw_box_landmarks_quality(orgimg,det,qs):
    '''
    画图 box,lands,quality
    :param orgimg:
    :param det: n*15
    :param qs:  n*1
    :return:
    '''
    img_box_lands=draw_box_landmarks(orgimg, det)
    for ind,q in enumerate(qs):
        # cv2.putText(img_box_lands,q,)
        x1,y1=det[ind,:2]
        lab='q:'+str(q.item())[:5]
        pt=(int(x1.item()), int(y1.item()) - 8)
        cv2.putText(img_box_lands, lab, pt, 0, fontScale=1.2, color=[225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        # cv2.putText(img0, lab, pt3, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=[225, 40, 168],
        #             thickness=2)

    return img_box_lands


# img_size = 640
# img_size=(1920,1920)
img_size=(288,512)
conf_thres = 0.3
iou_thres = 0.5

# model_path='/home/xiancai/classification-pytorch/detect_util/yolov5s-face_19201920_sim.onnx'
model_path='/home/xiancai/face_detection/yolov5-face/Result/2022_01_17/yolov5s-face_sim_288_512.onnx'
# model_path = '/home/xiancai/baby/yolov5/results/other/yolov5n-0.5.onnx' # (288,512)
ses=ort.InferenceSession(model_path)
print(f'face detection model: {model_path}')
if __name__ == '__main__':
    '''
    人脸检测，检测一个文件夹的图片
    '''


    input_path='/data1/xiancai/FACE_ANGLE_DATA/test/'
    sub_dir='_'.join(model_path.split('/')[-2:])+'-'+'_'.join(input_path.split('/')[-3:-1])+f'-imgsize{img_size}_conf{conf_thres}_iou{iou_thres}'
    res_path=f'/data1/xiancai/FACE_ANGLE_DATA/res_detect/{sub_dir}/'

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    ls=os.listdir(input_path)
    for ind, i in enumerate(ls):
        orgimg = cv2.imread(input_path+i)  # BGR
        t0=time_synchronized()
        det=detect_one_onnx(orgimg)
        t1 = time_synchronized()
        # 画图
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]] # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
        for j in range(det.shape[0]):
            xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist() # to xywh 原图ratio
            conf = det[j, 4].cpu().numpy()
            landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist() # to 原图ratio
            class_num = det[j, 15].cpu().numpy()
            orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
        # save
        cv2.imwrite(f'{res_path}res_{i}',orgimg)
        print(f'{ind}/{len(ls)}: saved to {res_path}res_{i}  {t1-t0}s')
    print('Done.')