'''
统计消防物品检测和分类精度
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


# 载入model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
# device=torch.device('cpu')
det_model_address='/home/xiancai/fire-equipment-demo/firecontrol_detection/Result/2021_11_25/best.pt'
# det_model_address='/home/xiancai/fire-equipment-demo/detect_classify/best.pt'
# model_detect = attempt_load(os.path.dirname(os.path.realpath(__file__)) + '/best.pt', map_location=device).eval()
model_detect = attempt_load(det_model_address, map_location=device).eval()
conf, iou = 0.35, 0.45  # NMS的阈值和iou 0.25 0.45

if __name__ == '__main__':
    # res = detect_classify('/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/images/val/53_d11_24_1637661828.jpg')
    # # res = detect_classify('/home/xiancai/Ruler/Pytorch/firecontrol_com/images/val/10_WIN_20201229_10_27_51_Pro.jpg')
    # print(res)


    '''
    测试检测+识别模型的精度 (检测一张图片，检测出的第一个类别与图片类别相同则为正确),错误图片保存在out_false_img
    测试一个文件夹内的全部图片, 文件名第一个前缀为类别  图片可以有0,1,多个obj,(多个obj测试第一个) 
    #图片有且只有一个obj
    #det_model_address   classify.model_address
    '''
    # init
    # root = '/home/xiancai/DATA/FIRE_DATA/fire_11_30/temp_jpg/'
    root = '/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/images/train/'
    print('---------------------------------------------------')
    print(f'test data: {root}\ntest det_model: {det_model_address}\ntest cls_model: {classify.cls_model_address}\n')

    ls=os.listdir(root)
    res = np.zeros((classify.cls_number+1, classify.cls_number+1), int)
    err_inf=[]
    err_imgs_addr=[] #错误图片地址
    err_pre=[]
    cls_ids = {v: k for k, v in classify.cls_names.items()}  # name:id

    # 检测+分类
    for ind,i in enumerate(ls):
        pre=detect_classify(root+i) #检测+分类 n*6 numpy
        # tru = classify.cls_names[int(i.split('_')[0]) - 1]  #

        pre_id = cls_ids[pre[0, -1]] if pre is not None else classify.cls_number
        fro=i.split('_')[0]#文件名第一个前缀为类别 图片有且只有一个类别
        tru =classify.cls_names[int(fro)] if fro!='f' else 'f'
        tru_id=int(fro) if fro!='f' else classify.cls_number

        if pre_id!=tru_id:
            err_imgs_addr.append(root+i)
            # err_inf.append(f'{root+i},tru:{tru},pre:{pre[0][-1]}')
            err_inf.append([root+i,tru_id,pre_id])
            # shutil.copy(root + i, out_false_img + i) #
        res[tru_id,pre_id]+=1
        print(ind, '/', len(ls),':', i, pre)
    # print(res)
    # for i in res:
    #     print(i)

    # 画res
    print(f'test data: {root}\ntest det_model: {det_model_address}\ntest cls_model: {classify.cls_model_address}\n')
    res_inf=[]
    for i,re in enumerate(res):
        tot=sum(re)
        acc=round(re[i]/sum(re),4) if sum(re) else 0.0
        tag= ''
        if acc<0.9: #
            tag='<-------'
            re_ind=sorted(range(len(re)),key=lambda x:re[x],reverse=True) #re降序后的下标
            # print(re_ind)
            for r_in in re_ind:
                r_ac=round(re[r_in]/tot,4) if tot!=0 else 0.0
                if r_ac > 0.01 and r_in!=i: #
                    tag+=f' {classify.cls_names[r_in]}:{r_ac}'
        r_inf=f'{i}:\ttot:{tot}\tacc:{acc} \t{classify.cls_names[i]} {tag}' #''.expandtabs(8)
        print(r_inf)
        res_inf.append([acc,r_inf])

    # 画依照acc排序的res
    res_inf_sort=sorted(range(len(res_inf)),key=lambda x:res_inf[x][0],reverse=True) #res_inf[][0]降序
    print('\nSORTED RES:')
    print(f'test data: {root}\ntest det_model: {det_model_address}\ntest cls_model: {classify.cls_model_address}\n')
    for res_inf_sor in res_inf_sort:
        print(res_inf[res_inf_sor][1])

    # for inf in err_inf:
    #     print(f'{i}err: ')
    print(f'err/total:{len(err_imgs_addr)}/{len(ls)}')
    aucc=round(1-len(err_imgs_addr)/len(ls),4)
    print(f'aucc:{aucc}')


    # 画框，保存错误图片err_imgs_addr
    dma, cma, rot = det_model_address.split('/'), classify.cls_model_address.split('/'), root.split('/')
    # out_false_img = '/home/xiancai/DATA/FIRE_DATA/detect/fal1'
    out_false_img=f'/home/xiancai/DATA/FIRE_DATA/detect/fal_{dma[-2]}_{dma[-1]}__{cma[-2]}_{cma[-1]}__{rot[-3]}_{rot[-2]}_{aucc}/' #检测错误图片存储路径
    if not os.path.exists(out_false_img):
        os.mkdir(out_false_img)

    for ad in err_imgs_addr:
        img0 = cv2.imread(ad)
        pre = detect_classify(ad)
        if pre is not None:
            # 画框
            for i in range(pre.shape[0]):
                H,W=img0.shape[:2]
                x,y,w,h=float(pre[i, 0]), float(pre[i, 1]),float(pre[i, 2]), float(pre[i, 3])
                x1,y1,x2,y2=int((x-w/2)*W),int((y-h/2)*H),int((x+w/2)*W),int((y+h/2)*H)
                pt1 = [x1, y1]
                pt2 = [x2,y2]
                cv2.rectangle(img0, pt1, pt2, (0, 255, 0), 2)

                pt3 = [x1 + 16, y1 + 16]
                lab = f'{round(float(pre[i, 4]), 2)}:{cls_ids[pre[i, 5]]}'
                cv2.putText(img0, lab, pt3, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=[225, 40, 168],
                            thickness=4)
        cv2.imwrite(out_false_img+'err_'+ad.split('/')[-1],img0)
    print(f'err img saved to {out_false_img}')

    from PIL import ImageDraw, ImageFont, Image


    # def img_writer(text_size, xy, text, text_color, fontStyle, img):
    #     draw = ImageDraw.Draw(img)
    #     # 字体的格式 这里的SimHei.ttf需要有这个字体
    #     fontStyle = ImageFont.truetype(fontStyle, text_size, encoding='utff8')
    #     # 绘制文本
    #     draw.text(xy, text, text_color, font=fontStyle)
    #
    #
    # if __name__ == '__main__':
    #     text_size = 50  # 字体大小
    #     xy = (56, 1700)  # 起始位置
    #     data = 'Menhenra酱，卡哇伊！'  # 内容
    #     text_color = (26, 42, 44)  # 字体颜色
    #     fontStyle = f'PingFang Bold.ttf'  # 字体位置
    #     img = Image.open('图片路径')
    #     img_writer(text_size, xy, data, text_color, fontStyle, img)