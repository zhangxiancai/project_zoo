'''
双层车牌推理 pt
'''
import os
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import cv2
import numpy as np
import torch
from collections import OrderedDict

from model.LPRNet import LPRNet_Double


class inference:


    model_path = '/home/xiancai/plate/LPRNet_Pytorch_Double/Result/2022_04_20_gener/best_1.0.pth'
    CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新', '学', '警', '挂',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'I', 'O', '-'
             ]
    CHARS = np.array(CHARS)
    CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

    # init model
    model_pt = LPRNet_Double(class_num=len(CHARS), dropout_rate=0, export=False).cpu()
    # load weights
    checkpoint = torch.load(model_path)
    state_dict_rename = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        state_dict_rename[name] = v
    model_pt.load_state_dict(state_dict_rename, strict=False)
    model_pt.eval()


    def infer(self,img_path):
        '''
        车牌识别接口 pt
        :param img_path:
        :return: 1*~ list int
        '''
        img=cv2.imread(img_path) # to h,w,c
        img=cv2.resize(img,(94,48)) # w=94,h=24
        # img=img[...,::-1] # to RGB
        img=img.transpose(2, 0, 1) # to 3*h*w
        img=img[None,...] # to 1*3*w*h
        img = np.ascontiguousarray(img).astype(np.float32) # to float
        img=(img-127.5)/128 # to [-1,1]

        # pre=ses.run(None,{'data':img})[0]
        # # print(pre.shape)
        pre=self.model_pt(torch.tensor(img)).detach().numpy() # 1*71*2*18
        pre=np.reshape(pre, [1,71,-1])

        pre=np.argmax(pre,axis=1)
        # res_lab = CHARS[pre]
        res= self.remove_repeate_blank_label(pre) # 删除重复和占位符 pre：['苏' 'E' '-' '-' '-' '6' '-' '-' '3' '-' '-' '7' '-' '-' 'A' '-' '-' 'F']
        # res_lab=CHARS[res]
        # print(res_lab)
        return res

    def draw(self,img_path,pre):
        img =cv2.imread(img_path)
        lab = ''.join(self.CHARS[pre])
        pt=[10,10]
        # draw lab
        img = self.paint_chinese_opencv(img, lab, pt, 20, color=(255, 255, 255))
        return img

    def remove_repeate_blank_label(self,pre):
        '''
        CTC:删除重复和占位符
        :param pre:  1*18 np
        :return:
        '''

        pre=pre[0].tolist()
        no_repeat_blank_label = list()
        pre.insert(0,len(self.CHARS)-1)
        p = pre[0]
        if p != len(self.CHARS) - 1:
            no_repeat_blank_label.append(p)
        for c in pre:  # dropout repeate label and blank label 注意
            if (c == p) or (c == len(self.CHARS) - 1): # 如果c与前一个字符相同或是占位符
                pass
            else:
                no_repeat_blank_label.append(c)
            p = c

        return no_repeat_blank_label

    def paint_chinese_opencv(self,im, chinese, position, fontsize, color):  # opencv输出中文
        img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 图像从OpenCV格式转换成PIL格式
        font = ImageFont.truetype('/home/xiancai/plate/simhei.ttf', fontsize, encoding="utf-8")
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, chinese,font=font, fill=color)  # PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
        return img

# def ttest():
#     '''
#     测试车牌识别模型pt，计算精度
#     测试一个文件夹的图片，图片名称格式 ~-川JK0707.jpg  或val.txt
#     错误图片保存至err_path
#     '''
#     # load data
#     imgs_path = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/train.txt'
#     DIR_TAG = False
#     if os.path.isdir(imgs_path):
#         ls = os.listdir(imgs_path)
#         DIR_TAG = True
#     else:
#         with open(imgs_path, 'r') as f:
#             ls = list(map(lambda x: x.strip(), f.readlines()))
#             # ls = ls[:1000]
#
#     # test
#     errs = []
#     tag = 0
#     for ind, i in enumerate(ls):
#         if not DIR_TAG:
#             ii = i.split('/')[-1]
#             imgs_path = i[:len(i) - len(ii)]
#             i = ii
#             # print(imgs_path)
#             # print(i)
#         t0 = time.time()
#         pre = classify_plate(imgs_path + i)  # classify
#         t1 = time.time()
#
#         # count
#         pre_lab = ''.join(CHARS[pre])
#         tru_lab = re.split('-|_', i)[-1][:-4].strip()  # ????????deepcamdata clean
#         if pre_lab != tru_lab:
#             errs.append((imgs_path, i, pre_lab))
#             tag += 1
#         print(f'{ind}/{len(ls)},inference:{t1 - t0}s')
#     acc = 1 - tag / len(ls)
#     print(f'acc:{acc}  {tag}/{len(ls)}')
#
#     # save
#     sub_dir = '_'.join(model_path.split('/')[-2:]) + '--' + '_'.join(imgs_path.split('/')[-3:-1]) + '--' + f'acc{round(acc, 4)}'
#     err_path = f'/data1/xiancai/PLATE_DATA/res_classify_err/{sub_dir}/'
#     if not os.path.exists(err_path):
#         os.mkdir(err_path)
#     for imgs_path, i, pre_lab in errs:
#         shutil.copy(imgs_path + i, err_path + i[:-4] + '-' + 'pre' + pre_lab + '.jpg')
#         print(f'err img saved to {err_path}')

class test():

    def __init__(self,engine):

        self.engine = engine
        self.DEBUG=True

    def test_img_muti(self,
                      imgs_glob='/data1/xiancai/PLATE_DATA/other/test/model_inference_result/*',
                      save_dir = '/data1/xiancai/PLATE_DATA/other/test/res_model_inference_result/'):
        '''
        检测+分类 多个图片
        :return:
        '''
        # img_glob='/data1/xiancai/PLATE_DATA/other/test/model_inference_result/*'
        # save_dir = '/data1/xiancai/PLATE_DATA/other/test/res_model_inference_result/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ls=glob.glob(imgs_glob)
        for index,i in enumerate(ls):
            pre=self.engine.infer(i) # 检测+分类
            img = self.engine.draw(i,pre)
            name = i.split('/')[-1]
            cv2.imwrite(f'{save_dir}{name}',img)
            print(f'{index}/{len(ls)}: saved to {save_dir}{name}')
        print('Done.')

    def test_video_one(self,
                       video_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/10A4BE72856C_monitoringOff_1618593172930.mp4',
                       save_path = '/data1/xiancai/BABY_DATA/other/test/Video2DeepCam/res_10A4BE72856C_monitoringOff_1618593172930.mp4'
                       ):
        '''
        检测视频,保存检测结果
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
        res = cv2.VideoWriter(save_path, mp4, fps/4, (w, h), True)  # WH
        numb = 0

        print(f'fps: {fps}, total: {total}, w: {w}, h: {h}')
        # 检测
        while (cap.isOpened()):
            numb += 1
            ret, frame = cap.read()
            # if numb % 2 == 0:
            if numb < total :
                if ret and numb%4==0:
                    print(f'{numb}/{total},frame.shape:{frame.shape}')
                    # detect
                    pre = self.engine.infer(frame) # 检测

                    # if pre.shape[0]==0: # 保存负样本
                    #     video_name=video_path.split('/')[-1]
                    #     record_save_path=f'/data1/xiancai/PLATE_DATA/driving_record_negative_imgs/{video_name}_{numb}.jpg'
                    #     cv2.imwrite(record_save_path,frame)
                    #     print(f'saved to {record_save_path}')

                    # draw
                    res_img = self.engine.draw(frame,pre)
                    print('')
                    # cv2.imwrite(f'{out_imgs}/{numb}_{name[:-4]}.jpg', res_img)  # video to frames
                    # if self.DEBUG:
                    #     cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/test/frame/{numb}.jpg',res_img)
                    res.write(res_img)  # 一帧保存至 mp4
                else:
                    continue
            else:
                break

        cap.release()
        res.release()
        print(f'saved to {save_path}')
        print('Done.')




if __name__=='__main__':


    test(inference()).test_img_muti('/data1/xiancai/PLATE_DATA/plate_double_generate/*',
                                                 save_dir='/data1/xiancai/PLATE_DATA/other/test_04_20/res_plate_double_generate/')