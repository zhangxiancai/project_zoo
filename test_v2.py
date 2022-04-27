# -*- coding: utf-8 -*-
"""
检测+根据landmarks计算人脸角度+计算人脸质量/
检测+6repNet计算角度+计算人脸质量
测试
"""
import glob
import os.path
import sys

import cv2
import math
import numpy as np

class base:

    def detect_onnx(self,img):
        pass


    def draw(self,img, pre):
        pass

class detection_blq(base):
    '''
    人脸检测 box+landmarks+quality
    '''
    sys.path.append('/home/xiancai/face_gender/classification-pytorch')
    from detect_util.detect_yolov5_face_onnx import detect_one_onnx,model_path

    def detect_onnx(self,img):
        '''
        获取box+landmarks+quality
        :param img:
        :return: pre
        '''
        DEBUG = True
        det = detection_blq.detect_one_onnx(img)
        # if DEBUG:
        #     img_box_lands=draw_box_landmarks(img,det)
        #     # cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/test/img_box_lands.jpg',img_box_lands)
        qs = []
        rpys = []
        det=det.numpy()
        for i in range(det.shape[0]):
            rotate_degree,quality=self.get_quality(det[i,:])
            qs.append(quality)
            rpys.append(rotate_degree)
            # print()
            # print(f'score:{score}, r:p:y: {rotate_degree}, quality:{quality}')
            # print(landmarks)
        # if DEBUG:
        #     img_box_lands_qs = detect_onnx_blq.draw_box_landmarks_quality(img, det, qs)
        #     cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/test/img_box_lands_qs2.jpg', img_box_lands_qs)
        # pre = (det,qs)
        cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/test/debug.jpg',img)
        pre = (det, qs, rpys)
        return pre

    def get_quality(self,pre,scale=1.0):
        '''
        根据landmarks计算人脸质量
        :param pre: 1*15 numpy  [x1x2y1y2,score,landmarks0-9]
        :return:
        '''

        (x1,y1,x2,y2),score,landmarks=pre[:4],pre[4],pre[5:15]
        leye,reye,nose,lmouth=landmarks[:2],landmarks[2:4],landmarks[4:6],landmarks[6:8]

        # yaw
        yaw_focus=2*(max(leye[0], reye[0], nose[0])-min(leye[0],reye[0],nose[0]))
        yaw_p1=-(leye[0]-nose[0])
        yaw_p2=reye[0]-nose[0]
        # if reye[0]-leye[0]<0:
        #     yaw=1
        # else:
        #     yaw=(yaw_p2-yaw_p1)/yaw_focus
        yaw = (yaw_p2 - yaw_p1) / yaw_focus

        # pitch
        pitch_focus=2*(max(lmouth[1],leye[1],nose[1])-min(lmouth[1],leye[1],nose[1]))
        pitch_p1 = -(leye[1]-nose[1])
        pitch_p2 = (lmouth[1] - nose[1]) * 1.1334
        pitch=(pitch_p2-pitch_p1)/pitch_focus

        # yaw, pitch加权和
        y = min(abs(yaw) * scale, 1.0)
        p = min(abs(pitch) * scale, 1.0) #放缩
        y_ratio = 1.0 - abs(y)
        p_ratio = 1.0 - abs(p)
        if abs(y_ratio - 1.0) < 0.0001 and abs(p_ratio - 1.0) < 0.0001:
            yp_ratio = 1.0
        else:
            y_w, p_w = (1 - y_ratio) / (1 - y_ratio + 1 - p_ratio), (1 - p_ratio) / (1 - y_ratio + 1 - p_ratio)  # y和p的权重
            yp_ratio = y_ratio * y_w + p_ratio * p_w

        # size of face box
        size = (y2 - y1) * (x2 - x1)
        size_ratio = min(size / (80 * 80), 1.0) #

        # quality
        # quality = (yp_ratio) * 0.45 + score * 0.45 + size_ratio * 0.1  # 阈值 80 80 ~
        yps_ratio=2*yp_ratio*score/(yp_ratio+score)
        quality = yps_ratio*0.9+size_ratio*0.1

        return (yaw,pitch),quality

    def draw(self,img, pre):
        '''
        画图 box,lands,quality
        :param orgimg:
        :param det: n*15 x1y1x2y2 conf land0-9
        :param qs:  n*1
        :return:
        '''
        # det,qs=pre
        det, qs, rpys= pre
        # img_box_lands = detect_onnx_blq.draw_box_landmarks(img, det)
        for ind, q in enumerate(qs):

            # draw quality and rpy
            x1, y1, x2, y2 = map(lambda x: int(x.item()), det[ind, :4])
            lab_q = f'q{round(q.item(),2)} '
            # lab_qry = f'rpy:{rpys[ind][0]},{rpys[ind][1]},{rpys[ind][2]}'
            lab_qry = f'yp{round(rpys[ind][0],2)},{round(rpys[ind][1],2)}'
            lab_score=f's{round(det[ind, 4].item(),2)}'
            # lab_score=''
            lab=lab_q+lab_qry
            # lab=lab_q+lab_score
            # pt = (int(x1.item()), int(y1.item()) - 8)
            pt = [x1, y1 - 8]
            # cv2.putText(img, lab, pt, 0, fontScale=1.2, color=[225, 255, 255], thickness=3,
            #             lineType=cv2.LINE_AA)
            cv2.putText(img, lab, pt, 0, fontScale=0.5, color=[0, 0, 225], thickness=1,
                        lineType=cv2.LINE_AA)

            # # draw rpy
            # lab_qry=f'rpy:{rpys[ind][0]},{rpys[ind][1]},{rpys[ind][2]}'
            # pt = [x1, y1 - 16]
            # cv2.putText(img, lab_qry, pt, 0, fontScale=1.2, color=[225, 255, 255], thickness=3,
            #             lineType=cv2.LINE_AA)

            # draw box
            if q>0.70:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # draw landmarks
            landmarks=list(map(lambda x:int(x.item()),det[ind, 5:]))
            for i in range(5):
                px,py=landmarks[2*i:2*i+2]
                cv2.circle(img, (px, py), 1, (0, 255, 0), -1)

        return img
class detection_repvgg(detection_blq):
    '''
    yoloface检测+repvgg识别角度
    '''
    sys.path.append('/home/xiancai/face_angle/6DRepNet/')
    from classify_pt import inference as repvgg_inference
    rep_inf=repvgg_inference()
    def detect_onnx(self,img):

        det = detection_blq.detect_one_onnx(img)
        qs = []
        rpys = []
        det = det.numpy()
        for i in range(det.shape[0]):
            # rotate_degree, quality = self.get_quality(det[i, :])
            x1,y1,x2,y2=map(int,det[i,:4])
            img_crop=img[y1:y2,x1:x2,...]
            p,y,r=self.rep_inf.classify(img_crop)
            p_ratio,y_ratio=max(1-abs(p)/90,0.001), max(1-abs(y)/90,0.001)
            quality=np.array(2*p_ratio*y_ratio/(p_ratio+y_ratio))
            qs.append(quality)
            rpys.append((y,p,r))

        cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/test/debug.jpg', img)
        pre = (det, qs, rpys)
        return pre



class test:
    '''
    测试
    '''
    def __init__(self,engine=detection_repvgg()):
        self.engine=engine # 用于 检测+分类
        self.DEBUG=True
        self.video_dir = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/'


    def test_video_one(self,
                       video_path = '/data1/xiancai/FACE_ANGLE_DATA/other/test/face_angle_test1.mp4',
                       save_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/res_10A4BE72856C_monitoringOff_1618593172930.mp4'
                       ):
        '''
        检测一个视频,保存检测结果
        :param video_path:
        :param save_path:
        :return:
        '''

        # 设置video读入与写出
        cap = cv2.VideoCapture(video_path)
        fps,total=cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 帧率，总帧数
        w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧宽，帧高
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        # res = cv2.VideoWriter(save_path, mp4, 20.0, (1280, 720), True)  # WH
        down_scale=1 # 下采样
        res = cv2.VideoWriter(save_path, mp4, fps/down_scale, (w, h), True)  # WH
        numb = 0

        print(f'fps: {fps}, total: {total}, w: {w}, h: {h}')
        # 检测
        while (cap.isOpened()):
            numb += 1
            ret, frame = cap.read()
            if numb % down_scale == 0:
                if ret:
                    print(f'{numb}/{total},frame.shape:{frame.shape}')
                    # detect
                    pre = self.engine.detect_onnx(frame)
                    # draw
                    res_img = self.engine.draw(frame,pre)
                    print('')
                    # cv2.imwrite(f'{out_imgs}/{numb}_{name[:-4]}.jpg', res_img)  # 一帧保存为图片
                    if self.DEBUG:
                        cv2.imwrite(f'/data1/xiancai/FACE_ANGLE_DATA/other/test/frames/{numb}.jpg',res_img)
                        save_frames_path=f'res_{video_path}_frames/'
                        if not os.path.exists(save_frames_path):
                            os.makedirs(save_frames_path)
                        cv2.imwrite(f'res_{video_path}_frames/{numb}.jpg', res_img)
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

    def test_img_one(self,img_path):
        rep_model_name=self.engine.rep_inf.pt_path.split('/')[-1]
        det_model_name=self.engine.model_path.split('/')[-1]
        img_name=img_path.split('/')[-1]
        save_path=img_path[:-len(img_name)]+rep_model_name+'_'+det_model_name+'_'+img_name
        img=cv2.imread(img_path)
        # detect
        pre = self.engine.detect_onnx(img)
        # draw
        res_img = self.engine.draw(img, pre)
        # save
        # name=img_path.split('/')[-1]
        # flag=len(img_path)-len(name)
        # cv2.imwrite(img_path[:flag]+'res_'+name,res_img)

        cv2.imwrite(save_path, res_img)
        print(f'saved to {save_path}')

    def test_img_mult(self,imgs_glob):
        # imgs_glob='/data1/xiancai/FACE_ANGLE_DATA/other/test_04_13/SideFace_OImages/*'
        ls= glob.glob((imgs_glob))
        for i in ls:
            self.test_img_one(i)



if __name__=='__main__':



    # test().test_video_one(video_path='/data1/xiancai/FACE_ANGLE_DATA/other/test/N25_01153854.111.mp4',save_path='/data1/xiancai/FACE_ANGLE_DATA/other/test/res_N25_01153854.111.mp4')
    #
    # test().test_img_one(img_path='/data1/xiancai/FACE_ANGLE_DATA/other/compare/微信图片_20220304170522.png')
    # # #
    # test().test_img_one(img_path='/data1/xiancai/FACE_ANGLE_DATA/other/compare/微信图片_20220304170515.jpg')
    #
    #
    #
    # test().test_img_one(img_path='/data1/xiancai/FACE_ANGLE_DATA/other/compare/微信图片_20220314184240.jpg',)
    #
    # test().test_img_one(img_path='/data1/xiancai/FACE_ANGLE_DATA/other/compare/微信图片_20220323140838.jpg')
    #
    # test().test_img_one(img_path='/data1/xiancai/FACE_ANGLE_DATA/other/test_04_13/0413_1.jpg')

    test().test_img_mult(imgs_glob='/data1/xiancai/FACE_ANGLE_DATA/other/test_04_13/SideFace_OImages/S*')
