'''
消防物品检测和分类
'''
import shutil

import torch
import os
import cv2
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
import classify


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
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
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# 载入model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
# device=torch.device('cpu')
det_model_address='/home/xiancai/fire-equipment-demo/firecontrol_detection/result/2021_12_06/best.pt'
# model_detect = attempt_load(os.path.dirname(os.path.realpath(__file__)) + '/best.pt', map_location=device).eval()
model_detect = attempt_load(det_model_address, map_location=device).eval()
conf, iou = 0.35, 0.45  # NMS的阈值和iou 0.25 0.45


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
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
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect(img_address):
    '''
    检测
    :param img_address:
    :return:[x,y,w,h,conf] (yolo格式)
    '''
    # 载入图片
    img = cv2.imread(img_address)
    H, W = img.shape[:2]
    # img = img[..., ::-1]  # BRG to RGB
    img = letterbox(img, 256, stride=32)[0]  # imgsize=256, stride=32
    # cv2.imwrite('预处理.jpg',img) # debug
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.unsqueeze(0)

    # detect
    pre = model_detect(img)[0]
    out = non_max_suppression(pre, conf_thres=conf, iou_thres=iou, classes=None, agnostic=False, labels=())[0]

    res = None
    # for item in out:
    if out.numel():
        box = scale_coords(img.shape[2:], out[:, :4],
                           [H, W, 3]).round()  # Rescale coords (xyxy) from img1_shape to img0_shape
        box = box.cpu().numpy()  # x1y1x2y2
        out = out.cpu().numpy()
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]  # 在原图上的坐标
        #

        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        xywh = np.vstack((x, y, w, h)).T / np.array([W, H, W, H])
        res = np.hstack((xywh, out[:, 4].reshape(box.shape[0], 1)))

        # point1 = (int(x1[0]), int(y1[0]))
        # point2 = (int(x2[0]), int(y2[0]))
        # test_img=cv2.imread(img_address)
        # cv2.rectangle(test_img, point1, point2, (0, 255, 0), 2)
        # cv2.imwrite('检测框.jpg', test_img)  # debug
    return res


def detect_classify(img_address):
    '''
    检测并分类
    :param img_address:
    :return: [xywh conf cls]
    '''
    # detect
    pre = detect(img_address)
    if pre is None:
        return None

    cls = []
    for i in range(pre.shape[0]):
        cl = classify.classify(img_address, pre[i, :4])  #
        cls += [cl]
    return np.hstack((pre, np.array(cls).reshape(len(cls), 1)))


if __name__ == '__main__':
    # res = detect_classify('/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/images/val/53_d11_24_1637661828.jpg')
    # # res = detect_classify('/home/xiancai/Ruler/Pytorch/firecontrol_com/images/val/10_WIN_20201229_10_27_51_Pro.jpg')
    # print(res)


    # 测试训练集上的全部图片
    # root='/home/xiancai/Ruler/Pytorch/firecontrol/images/train/'
    root = '/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/images/val/'
    out_false_img='/home/xiancai/DATA/FIRE_DATA/detect/fal1/' #检测错误图片存储路径
    if not os.path.exists(out_false_img):
        os.mkdir(out_false_img)
    ls=os.listdir(root)

    cls_ids={v:k for k,v in classify.cls_names.items()} # name:id
    res = np.zeros((classify.cls_number, classify.cls_number), int)
    err=[]
    for ind,i in enumerate(ls):
        pre=detect_classify(root+i) #检测+分类 n*6 numpy
        # tru = classify.cls_names[int(i.split('_')[0]) - 1]  #
        tru = classify.cls_names[int(i.split('_')[0]) ]  #文件名第一个前缀为类别 图片有且只有一个类别

        #
        if pre is None:
            err.append(f'{root+i},tru:{tru},pre:{pre}')
            shutil.copy(root+i,out_false_img+i)
            continue
        pre_id,tru_id=cls_ids[pre[0, -1]], cls_ids[tru] #只取 pre[0]
        if pre_id!=tru_id:
            err.append(f'{root+i},tru:{tru},pre:{pre}')
            shutil.copy(root + i, out_false_img + i)
        res[tru_id,pre_id]+=1
        print(ind, '/', len(ls), i, pre)
    print(res)

    # 画res
    for i,re in enumerate(res):
        tot=sum(re)
        acc=round(re[i]/sum(re),4) if sum(re) else 0.0
        tag='<------' if acc<1 else ''
        print(f'{i}:\ttotal:{tot}\tacc:{acc} \t{classify.cls_names[i]} {tag}')

    for i in err:
        print(f'err: {i}')
    print(f'err/total:{len(err)}/{len(ls)}')
    print(f'aucc:{1-len(err)/len(ls)}')