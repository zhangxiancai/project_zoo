'''
测试 onnxruntime
'''

import glob
import os
import time

import onnx
from onnx import shape_inference
import onnxruntime as ort
import numpy as np
import cv2
import torch
import torchvision
# import google.protobuf.pyext._message.RepeatedCompositeContainer

class detect_onnx:
    '''

    '''
    # model_address='/home/xiancai/Ruler/result/2021_12_15/ruler_12_15_sim.onnx'
    # model_address = '/home/xiancai/baby/yolov5/results/2022_02_22/baby_best_320_320.onnx'
    # model_address = '/home/xiancai/baby/yolov5/results/2022_03_18_n0.5/n0.5_192_320.onnx'
    # model_address='/home/xiancai/Ruler/result/2021_10_18/best_10_18.onnx'
    # model_address='/home/xiancai/test/best0.onnx'
    # model_address = '/home/xiancai/baby/yolov5/results/2022_03_03/best.onnx'
    model_address = '/home/xiancai/baby/yolov5/results/2022_05_27_blazeSingleGray/baby_blazeSingleGray_05_27.onnx'

    ses = ort.InferenceSession(model_address)

    # get hw of onnx input
    model = onnx.load(model_address)
    input_h = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    input_w = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value # onnx input: bchw
    input_channel = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value # 输入信道
    out_counts = len(model.graph.output) # 输出个数
    # net=onnx.load(model_address)
    # onnx.checker.check_model(net)
    # print(onnx.helper.printable_graph(net.graph))
    # net=cv2.dnn.readNet(model_address)
    # net.setInput(blob)
    # outs = net.forward(net.getUnconnectedOutLayersNames())

    # conf=0.35
    conf = 0.35
    iou = 0.45 # 0.45
    cls_name=['baby']
    cls = 1
    if out_counts==1:
        anchors=[55, 72, 225, 304, 438, 553]
    else:
        # anchors = [[28, 63, 86, 126, 78, 224], [89, 344, 74, 621, 362, 146], [391, 340, 374, 637, 426, 641]]
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        '''
        Resize and pad image while meeting stride-multiple constraints
        :param img:
        :param new_shape:
        :param color:
        :param auto:  True: minimum rectangle
        :param scaleFill:
        :param scaleup:
        :param stride:
        :return:
        '''

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

        # resize and pad
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results
            prediction 1*~*17 xywh obj cls
        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence  obj>conf *************

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf 相乘

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1) # obj*cl_con>conf then convert to [ box,obj*cl_con,c_id ]
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1  tensor:clamp_
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def detect_onnx(self, img_address):
        '''
        onnx检测
        :param img_address: 单个图片地址 or ndarray : h*w*c
        :return: pre: n*(x1,y1,x2,y2,conf,cls) numpy
        '''

        #预处理
        img0 = cv2.imread(img_address) if isinstance(img_address,str) else img_address

        # if DEBUG:
        #     img0=img0.transpose(1,0,2) #debug
        img = self.letterbox(img0, (self.input_h,self.input_w),auto=False, stride=32)[0]  # Padded resize
        cv2.imwrite('/data1/xiancai/BABY_DATA/other/debug.jpg',img)
        if self.input_channel==1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[None,...]
        else:
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = img[None,...]

        # interference (1*~*~*51)*3
        outs=self.ses.run(None,{'images':img})
        # self.net.setInput(img)
        # outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # onnx 输出转换
        anchors = self.anchors
        grid = [torch.zeros(1)] * 3 # 3 layer
        if self.out_counts==1: # blazeSingle
            stride = torch.tensor([16.0])
        else:
            stride=torch.tensor([8.0,16.0,32.0])
        a = torch.tensor(anchors).float().view(3, -1, 2)# 3 layer
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2) # 3 layer

        z=[]
        for i in range(self.out_counts):
            # outs[i] to x 1*3*h*w*c
            if outs[i].shape[-1]==3*(self.cls+5): #如果为[bhw3c]*3
                x=outs[i].reshape(1,outs[i].shape[1],outs[i].shape[2],3,self.cls+5) # 1*~*~*3*17
                x=torch.tensor(x).permute(0,3,1,2,4).contiguous() # 1*3*~*~*17
            else:
                x = outs[i].reshape(1, 3, self.cls + 5, outs[i].shape[2], outs[i].shape[3])
                x = torch.tensor(x).permute(0, 1, 3, 4, 2).contiguous()

            # nx,ny=x.shape[2],x.shape[3]
            ny, nx = x.shape[2], x.shape[3]
            if grid[i].shape[2:4] != x.shape[2:4]:
                grid[i] = self.make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x.device)) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(1, -1, self.cls+5)) # 17
        z=torch.cat(z,1) # 1*~*17
        # print(f'outs to z: \nz_shape:{z.shape},z[0,0,...]:{z[0,0,...]}')

        # NMS
        pre=self.non_max_suppression(z,conf_thres=self.conf,iou_thres=self.iou)
        # print(f'pre:{pre}')
        pre=pre[0].numpy() # 1张图片

        # x1y1x2y2 to 原图像素
        pre[:, :4] = self.scale_coords(img.shape[2:], pre[:, :4], img0.shape).round()  #
        return pre

    def draw_box(self,img_address,pre):
        '''
        画框
        :param img_address: 单个图片地址 or ndarray : h*w*c
        :param pre: n*(x1,y1,x2,y2,conf,cls) numpy
        :return:
        '''

        img0 = cv2.imread(img_address) if isinstance(img_address,str) else img_address

        for i in range(pre.shape[0]):
            # draw box
            pt1 = [int(pre[i, 0]), int(pre[i, 1])]
            pt2 = [int(pre[i, 2]), int(pre[i, 3])]
            cv2.rectangle(img0, pt1, pt2, (0, 255, 0), 2)
            # draw lab
            pt3 = [int(pre[i, 0]) + 16, int(pre[i, 1]) + 16]
            lab = f'{str(np.round(pre[i, 4], 4))}:{self.cls_name[int(pre[i, 5])]}'
            cv2.putText(img0, lab, pt3, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=[225, 40, 168],
                        thickness=2)
        return img0



class test(detect_onnx):
    '''
    测试
    '''

    DEBUG=False
    video_dir = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/'


    def test_video_one(self,
                       video_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/10A4BE72856C_monitoringOff_1618593172930.mp4',
                       save_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/res_10A4BE72856C_monitoringOff_1618593172930.mp4'
                       ):
        # 检测视频,保存检测结果

        # 设置输入输出视频路径
        # video_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/10A4BE72856C_monitoringOff_1618593172930.mp4'
        # save_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/res_10A4BE72856C_monitoringOff_1618593172930.mp4'
        # name = in_video.split('/')[-1]
        # out_video = f'{out_iv}/res_{name[:-4]}.mp4'
        # out_imgs = f'{out_iv}/{name[:-4]}'
        # print(f'out_video:{out_video}, out_imgs:{out_imgs}')
        # if not os.path.exists(out_imgs):
        #     os.mkdir(out_imgs)

        # 设置video读入与写出
        cap = cv2.VideoCapture(video_path)
        fps,total=cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 帧率，总帧数
        w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧宽，帧高
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        # res = cv2.VideoWriter(save_path, mp4, 20.0, (1280, 720), True)  # WH
        res = cv2.VideoWriter(save_path, mp4, fps/10, (w, h), True)  # WH
        numb = 0

        print(f'fps: {fps}, total: {total}, w: {w}, h: {h}')
        # 检测
        while (cap.isOpened()):
            numb += 1
            ret, frame = cap.read()
            if numb % 50 == 0:
                if ret:
                    print(f'{numb}/{total},frame.shape:{frame.shape}')
                    # detect
                    pre = self.detect_onnx(frame)
                    # draw
                    res_img = self.draw_box(frame,pre)
                    print('')
                    # cv2.imwrite(f'{out_imgs}/{numb}_{name[:-4]}.jpg', res_img)  # 一帧保存为图片
                    if self.DEBUG:
                        cv2.imwrite(f'/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/1/{numb}.jpg',res_img)
                    res.write(res_img)  # 一帧保存至 mp4
                else:
                    break

        cap.release()
        res.release()
        print('Done.')

    def test_video_mult(self):
        ls=glob.glob(self.video_dir+'*')
        for ind, i in enumerate(ls):
            print(f'{ind}/{len(ls)} video:')
            save_path='/'.join(i.split('/')[:-1])+'/res/320320_320320_res_'+i.split('/')[-1]
            self.test_video_one(i,save_path)

    def test_img_one(self,img_path,save_path):
        img=cv2.imread(img_path)
        # detect
        pre = self.detect_onnx(img)
        # draw
        res_img = self.draw_box(img, pre)
        # save
        cv2.imwrite(save_path, res_img)  #


if __name__=='__main__':

    test().test_img_one(img_path='/data1/xiancai/BABY_DATA/other/test_05_20/微信图片_20220523181044.jpg',save_path='/data1/xiancai/BABY_DATA/other/test_05_20/res_微信图片_20220523181044.jpg')