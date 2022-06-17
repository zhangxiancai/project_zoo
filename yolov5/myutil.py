'''
数据处理脚本
'''
import glob
import json
import os.path
import random
import shutil

import cv2
import numpy as np


class base:
    '''
    神目格式 to yolo格式
    '''

    def __init__(self):
        pass

    def single_jsontotxt(self, json_dir, out_dir):
        '''
        json转txt，单个文件，无landmark，不保存adult标签，只保存baby
        :param json_dir:输入json文件地址
        :param out_dir:输出txt文件地址  cls xywh
        :return:
        '''

        # load json
        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)

        # each box
        file_str = ''
        for t in content['faces']:

            # json越界修正
            x1, x2, y1, y2 = t['x'], t['x'] + t['w'], t['y'], t['y'] + t['h']
            W, H = content['image_width'], content['image_height']
            x1, x2 = max(x1, 0), min(x2, W)
            y1, y2 = max(y1, 0), min(y2, H)
            if x1 > W or x2 < 0 or y1 > H or y2 < 0:
                print(f'json label error: {json_dir} x1 {x1},y1 {y1},x2 {x2},y2 {y2}')
                raise

            # convert to xywh
            x = (x1 + (x2 - x1) / 2) / W
            y = (y1 + (y2 - y1) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H

            # type
            type = 0  # 只检测

            # cat
            if t['type'] == 'Baby':  # 只保存baby
                # file_str += str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
                #     round(w, 6)) + ' ' + str(round(h, 6))
                file_str += str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
                    round(w, 6)) + ' ' + str(round(h, 6)) + '\n' # 4月12日更改

        # save
        filename = out_dir
        if os.path.exists(filename):
            os.remove(filename)
        os.mknod(filename)  #
        fp = open(filename, mode="r+", encoding="utf-8")
        fp.write(file_str[:-1])
        fp.close()

    def make_jpg_xml_from_origin(self):
        if not os.path.exists(self.jpg):
            os.mkdir(self.jpg)
        if not os.path.exists(self.xml):
            os.mkdir(self.xml)
        ls = glob.glob(self.glob_str)
        for ind, i in enumerate(ls):
            if i[-3:] in ['jpg', 'jpeg', 'png']:
                shutil.copy(i, self.jpg)
            if i[-3:] == 'xml':
                shutil.copy(i, self.xml)
            print(f'{ind}/{len(ls)}')

    def make_txt_from_xml(self):
        '''
        xml to txt
        '''

        if not os.path.exists(self.txt):
            os.mkdir(self.txt)
        ls = glob.glob(self.xml + '*')
        total = 0
        for i in ls:
            if i[-3:] == 'xml':
                nam = i.split('/')[-1]
                txt_fi = f'{self.txt}{nam[:-7]}txt'
                self.single_jsontotxt(i, txt_fi)  # 转换
                total += 1
                print(f'{total}/{int(len(ls))}: saved to {txt_fi}')
        print(f'saved to {self.txt}')

    # def make_yolov5(self):
    #     pass

    def update_yolov5(self):
        '''
        make yolov5 or add a unit to yolov5
        '''

        # source
        self.info()

        # dst
        out_fir_tra_img = self.yolov5 + 'images/train/'
        out_fir_val_img = self.yolov5 + 'images/val/'
        out_fir_tra_txt = self.yolov5 + 'labels/train/'
        out_fir_val_txt = self.yolov5 + 'labels/val/'
        iltv_dir = [out_fir_tra_img, out_fir_val_img, out_fir_tra_txt, out_fir_val_txt]
        for i in iltv_dir:
            if not os.path.exists(i):
                os.makedirs(i)

        print('before:')
        t_i, v_i, t_tx, v_tx = len(os.listdir(out_fir_tra_img)), len(os.listdir(out_fir_val_img)), \
                               len(os.listdir(out_fir_tra_txt)), len(os.listdir(out_fir_val_txt))
        print(f'{out_fir_tra_img}:{t_i}')
        print(f'{out_fir_val_img}:{v_i}')
        print(f'{out_fir_tra_txt}:{t_tx}')
        print(f'{out_fir_val_txt}:{v_tx}')

        # 划分 各类随机抽1%
        ls_img = glob.glob(self.jpg + '*')
        random.shuffle(ls_img)
        flag = max(int(0.01 * len(ls_img)) + 1, 1)  # 1%
        ls_img_val = ls_img[:flag]
        ls_img_tra = ls_img[flag:]

        # 添加至验证集
        for i in ls_img_val:
            # jpg
            nam = i.split('/')[-1]
            shutil.copy(i, out_fir_val_img + f'{self.buff}_{nam}')  # 添加至val_img
            # lab
            txt_abpath = self.txt + nam[:-4] + '.txt'  #
            if os.path.exists(txt_abpath):  # 如果有标签文件
                shutil.copy(txt_abpath, out_fir_val_txt + f'{self.buff}_{nam[:-4]}.txt')

        # 添加至训练集
        for ind, i in enumerate(ls_img_tra):
            # jpg
            nam = i.split('/')[-1]
            shutil.copy(i, out_fir_tra_img + f'{self.buff}_{nam}')  # 添加至tra_img
            # lab
            txt_abpath = self.txt + nam[:-4] + '.txt'
            if os.path.exists(txt_abpath):
                shutil.copy(txt_abpath, out_fir_tra_txt + f'{self.buff}_{nam[:-4]}.txt')  # 添加至tra_lab

        # check
        print('after:')
        t_i_d, v_i_d, t_tx_d, v_tx_d = len(os.listdir(out_fir_tra_img)), len(os.listdir(out_fir_val_img)), \
                                       len(os.listdir(out_fir_tra_txt)), len(os.listdir(out_fir_val_txt))
        print(f'{out_fir_tra_img}:{t_i_d}')
        print(f'{out_fir_val_img}:{v_i_d}')
        print(f'img added: {t_i_d + v_i_d - t_i - v_i}')
        print(f'{out_fir_tra_txt}:{t_tx_d}')
        print(f'{out_fir_val_txt}:{v_tx_d}')
        print(f'lab added: {t_tx_d + v_tx_d - t_tx - v_tx}')

    def remove_yolov5(self, buff):
        '''
        remove a unit from yolov5
        :return:
        '''
        ls = glob.glob(f'{self.yolov5}*/*/*')
        for ind, i in enumerate(ls):
            if i.split('/')[-1][:len(buff)] == buff:  # 根据前缀删除
                os.remove(i)
                print(f'{ind}/{len(ls)}:{i}')

    def info(self):
        ls = glob.glob(self.glob_str)
        print(self.buff)
        print(
            f'origin:{len(ls)}\njpg:{len(os.listdir(self.jpg))}\nxml:{len(os.listdir(self.xml))}\ntxt:{len(os.listdir(self.txt))}\n')

    def info_txt(self):
        pass


class deepcam_baby(base):
    '''
    神目格式的baby图片
    '''

    def __init__(self):
        self.root = '/data1/xiancai/BABY_DATA/2022_02_19/'
        self.buff = f'd02_19'  # 前缀 unit
        # self.origin=f'{self.root}origin/'
        self.glob_str = f'{self.root}origin/*/*'

        self.jpg = f'{self.root}jpg/'
        self.xml = f'{self.root}xml/'
        self.txt = f'{self.root}txt/'

        self.yolov5 = f'/data1/xiancai/BABY_DATA/baby_detect_yolov5/'  # yolov5数据集地址


class ruler_03_17(base):
    '''

    '''
    def __init__(self):
        self.root = '/home/xiancai/DATA/RULER_DATA/2022_03_17/'
        self.buff = f''  # 前缀 unit
        self.glob_str = f'{self.root}origin_pre/*'

        self.jpg = f'{self.root}jpg/'
        self.xml = f'{self.root}xml/'
        self.txt = f'{self.root}txt/'

        self.yolov5 = f'/home/xiancai/DATA/RULER_DATA/2022_03_17/ruler_yolov5_03_17/'  # yolov5数据集地址


class d2022_03_03(base):
    glob_str = '/data1/xiancai/BABY_DATA/2022_03_03/bayb下载图片/布娃娃 玩偶/*/*/*'
    jpg = '/data1/xiancai/BABY_DATA/2022_03_03/jpg/'
    xml = '/data1/xiancai/BABY_DATA/2022_03_03/xml/'
    txt = '/data1/xiancai/BABY_DATA/2022_03_03/txt/'
    yolov5 = '/data1/xiancai/BABY_DATA/baby_detect_yolov5_03_03/'  # yolov5数据集地址
    buff = 'd03_03_f'

    def run(self):
        # self.make_jpg_xml_from_origin()
        # self.make_txt_from_xml()
        self.update_yolov5()
        # self.info()

class d2022_03_18(base):
    glob_str = '/data1/xiancai/BABY_DATA/2022_03_18/origin_pre/*'
    jpg = '/data1/xiancai/BABY_DATA/2022_03_18/jpg/'
    xml = '/data1/xiancai/BABY_DATA/2022_03_18/xml/'
    txt = '/data1/xiancai/BABY_DATA/2022_03_18/txt/'
    yolov5 = '/data1/xiancai/BABY_DATA/baby_detect_yolov5_03_18_xx/'  # yolov5数据集地址
    buff = 'd03_18'

    def run(self):
        # self.make_jpg_xml_from_origin()
        self.make_txt_from_xml()
        self.update_yolov5()
        self.info()

class d2022_05_17(base):
    glob_str = '/data1/xiancai/BABY_DATA/2022_05_17/baby_detection_20220425/*'
    jpg = '/data1/xiancai/BABY_DATA/2022_05_17/jpg/'
    xml = '/data1/xiancai/BABY_DATA/2022_05_17/xml/'
    txt = '/data1/xiancai/BABY_DATA/2022_05_17/txt/'
    yolov5 = '/data1/xiancai/BABY_DATA/baby_detect_yolov5_05_17/'  # yolov5数据集地址
    buff = 'd05_17'

    def run(self):
        self.make_jpg_xml_from_origin()
        self.make_txt_from_xml()
        self.update_yolov5()
        self.info()

class video2imgs:
    '''
    从视频截帧 做为训练测试集
    '''
    down_scale = 32  # 下采样
    thd = 0.1  # 超过20%像素变化则保存帧
    di = 2  # 像素值差大于di则认为变化
    DEBUG = False
    frames_tag = 0  # 保存的帧数

    def get_diff(self, fra1, fra2):
        '''
        计算相邻两帧的不同像素比例
        :param fra1:
        :param fra2:
        :return:
        '''
        if fra1.size == 0 or fra2.size == 0:
            return 1
        fra1 = cv2.cvtColor(fra1, cv2.COLOR_BGR2GRAY)  # h*w*c to h*w
        fra2 = cv2.cvtColor(fra2, cv2.COLOR_BGR2GRAY)
        diff_matrix = np.abs(fra1 - fra2)

        # diff=np.sum(diff_matrix)/(fra1.shape[0]*fra1.shape[1])
        diff = np.sum(diff_matrix > self.di) / (fra1.shape[0] * fra1.shape[1])
        return diff

    def get_diffs_video_one(self,
                            video_path='/data1/xiancai/BABY_DATA/2022_03_10/origin/2Deep/E062904F3842_monitoringOff_1637315193824.mp4', ):
        '''

        :param video_path:
        :return:
        '''
        # 设置video读入
        cap = cv2.VideoCapture(video_path)
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧率，总帧数
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧宽，帧高
        down_scale = self.down_scale  # 下采样

        numb = 0
        print(f'fps: {fps}, total: {total}, w: {w}, h: {h}')

        # per frame
        fra1 = np.empty((0, 0))
        fra2 = np.empty((0, 0))
        diffs = []
        while (cap.isOpened()):
            numb += 1
            ret, frame = cap.read()
            if numb % down_scale == 0:
                if ret:
                    fra1 = fra2
                    fra2 = frame.copy()
                    diff = self.get_diff(fra1, fra2)
                    diffs.append(diff)
                else:
                    break
        cap.release()
        return diffs

    def video2imgs_one(self,
                       video_path='/data1/xiancai/BABY_DATA/2022_03_10/origin/2Deep/E062904F3842_monitoringOff_1637315193824.mp4',
                       save_dir='/data1/xiancai/BABY_DATA/other/test/debug_frame/'
                       ):
        '''
        single video to imgs
        :param video_path:
        :param save_path:
        :return:
        '''

        diffs = self.get_diffs_video_one(video_path)
        if len(diffs) == 0:
            return
        mean_diffs = sum(diffs) / len(diffs)
        # 设置video读入
        cap = cv2.VideoCapture(video_path)
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧率，总帧数
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧宽，帧高
        down_scale = self.down_scale  # 下采样

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        res = cv2.VideoWriter(save_dir + 'debug.mp4', mp4, fps / down_scale, (w, h), True)  # WH
        numb = 0
        print(f'fps: {fps}, total: {total}, w: {w}, h: {h}')

        # per frame
        fra1 = np.empty((0, 0))
        fra2 = np.empty((0, 0))
        name = video_path.split('/')[-1]
        while (cap.isOpened()):
            numb += 1
            ret, frame = cap.read()

            if numb % down_scale == 0:

                if ret:
                    print(f'{numb}/{total},frame.shape:{frame.shape},m_d:{mean_diffs},saved frames:{self.frames_tag}')
                    fra1 = fra2
                    fra2 = frame.copy()
                    diff = self.get_diff(fra1, fra2)

                    # save
                    if diff > self.thd + mean_diffs:
                        cv2.imwrite(f'{save_dir}/{name}_{numb}.jpg', frame)
                        self.frames_tag += 1

                    if self.DEBUG:
                        # save to mp4
                        lab = f'diff:{diff}'
                        pt = [100, 100]
                        cv2.putText(frame, lab, pt, 0, fontScale=0.5, color=[225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        res.write(frame)
                else:
                    break

        cap.release()
        res.release()
        print('Done.')

    def video2imgs_muti(self):
        glob_str = '/data1/xiancai/BABY_DATA/2022_03_10/origin/*/*'  # videos
        save_dir = '/data1/xiancai/BABY_DATA/2022_03_10/jpg_32_m0.1/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        ls = glob.glob(glob_str)
        for ind, i in enumerate(ls):
            print(f'{ind}/{len(ls)}:{i}')
            self.video2imgs_one(i, save_dir)

    def get_half(self):
        glob_str = '/data1/xiancai/BABY_DATA/2022_03_10/jpg/*'
        ls = glob.glob(glob_str)
        for ind, i in enumerate(ls):
            if ind % 3 == 0 or ind % 3 == 1:
                os.remove(i)


if __name__ == '__main__':
    # deepcam_baby().remove_yolov5(buff='d03_03')

    # video2imgs().video2imgs_muti()
    # video2imgs().get_half()

    d2022_03_18().run()
