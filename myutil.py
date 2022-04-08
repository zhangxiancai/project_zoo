import glob
import os
import random
import re
import shutil

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image

import rep_utils

class Pose_300W_LP:
    '''

    '''

    def draw_img_mat_one(self, img_path='', mat_path='', save_path=''):
        '''
        将mat注释画到图片上
        :param img_path:
        :param mat_path:
        :param save_path:
        :return:
        '''
        # img_path='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/image00047.jpg'
        # mat_path='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/AFLW2000/image00047.mat'
        # save_path='/data1/xiancai/FACE_ANGLE_DATA/other/debug/1.jpg'

        # img_path='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/AFW/AFW_1051618982_1_17.jpg'
        # mat_path='/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/AFW/AFW_1051618982_1_17.mat'
        # save_path='/data1/xiancai/FACE_ANGLE_DATA/other/debug/AFW_1051618982_1_17.jpg'

        img = cv2.imread(img_path)
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        pos = mat['Pose_Para'][0]  # [pitch yaw roll tdx tdy tdz scale_factor]
        pyr = pos[:3]  # radians

        # draw pt2d
        for i in range(pt2d.shape[1]):
            pt = (int(pt2d[0, i]), int(pt2d[1, i]))
            cv2.circle(img, pt, 1, [0, 255, 0], thickness=5)

        # draw pyr
        pyr_degree = pyr * 180 / np.pi
        lab = f'pyr:{pyr_degree.astype(np.int32)}'
        cv2.putText(img, lab, (100, 100), 0, 1, [0, 0, 255], thickness=3)
        # save
        # cv2.imwrite(save_path,img)
        return img

    def draw_multi(self):

        glob_str = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/*/*.jpg'
        jpgs = glob.glob(glob_str)
        random.shuffle(jpgs)

        # draw 16 imgs
        imgs = []
        for ind, i in enumerate(jpgs[:16]):
            print(i)
            img_path = i
            mat_path = i[:-4] + '.mat'

            img = self.draw_img_mat_one(img_path, mat_path)
            imgs.append(img)

        # cat 4*4
        imgs4 = []
        for flag in range(4):  #
            imgs4.append(np.concatenate(imgs[4 * flag:4 * flag + 4], axis=1))
        big_img = np.concatenate(imgs4, axis=0)
        cv2.imwrite('/data1/xiancai/FACE_ANGLE_DATA/other/debug/AFLW16.jpg', big_img)

    def make_txt(self):
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/*/*.jpg'
        txt = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP_crop/train.txt'

        ls = glob.glob(imgs_glob)
        # content = '\n'.join(map(lambda x: x +'  '+ '  '.join(map(lambda x: str(round(x*180/np.pi,4)), rep_utils.get_ypr_from_mat(x[:-3]+'mat'))), ls))

        content = '\n'.join(map(lambda x: '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP_crop/'+x.split('/')[-1] + '  ' + '  '.join(
            map(lambda x: str(round(x * 180 / np.pi, 4)), rep_utils.get_ypr_from_mat(x[:-3] + 'mat'))), ls)) # 间隔两个空格

        # save
        with open(txt,'w') as f:
            f.write(content)
        print(f'saved to {txt}')

    def crop(self):
        '''
        因为图片背景占比大, 扣图 (根据mat中的kp)
        :return:
        '''
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/*/*.jpg'
        save_dir = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP_crop/'
        ls = glob.glob(imgs_glob)
        for ind,i in enumerate(ls):
            img = Image.open(i)
            mat_path = i[:-3]+'mat'
            pt2d = rep_utils.get_pt2d_from_mat(mat_path)  #
            pose = rep_utils.get_ypr_from_mat(mat_path)  ##We get the pose in radians
            # print(f'pt2d:{pt2d}')
            # print(f'pose{pose}')

            # crop
            x_min = min(pt2d[0, :])
            y_min = min(pt2d[1, :])
            x_max = max(pt2d[0, :])
            y_max = max(pt2d[1, :])
            k = np.random.random_sample() * 0.2 + 0.2  # k = 0.2 to 0.40
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # p,y,r = map(lambda x:str(round(x*180/np.pi,2)),pose[:3]) # 注意顺序

            save_path = save_dir + i.split('/')[-1]
            img.save(save_path)
            print(f'{ind}/{len(ls)}:saved to {save_path}')



    def check(self):
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_15/300W_LP/*/*.jpg'
        ls = glob.glob(imgs_glob)
        for i in ls:
            if '  ' in i:
                print(i)

# class dataset_info:
#     '''
#     查看数据集信息
#     '''

# def convert_img():
#     '''
#     转换图片，用于海思量化（不同标准差）
#     :return:
#     '''
#     std=[0.229, 0.224, 0.225]
#     mean=[0.485, 0.456, 0.406]
#     imgs_glob='/data1/xiancai/FACE_ANGLE_DATA/other/wk/images/000210871White2.jpg'
#     save_dir='/data1/xiancai/FACE_ANGLE_DATA/other/wk/res_images/'
#     ls = glob.glob(imgs_glob)
#     for i in ls:
#
#         img=cv2.imread(i)
#         img = img[...,::-1] # to rgb
#         #
#         imgwk=np.zeros((112,112,3),dtype=np.float32)
#         for c in range(3):
#             imgwk[...,c]= img[...,c]/std[c]+255*mean[c]*(1-1/std[c])
#
#         # img_name=i.split('/')[-1]
#         # cv2.imwrite(f'{save_dir}0.tiff',img)
#
#         mea = np.array(mean).reshape(1,1,3)
#         st = np.array(std).reshape(1, 1, 3)
#         imgfinal=imgwk/255-mean
#
#         imgpc=img.copy()
#         imgpc=(imgpc/255-mea)/st


class Web_collection:
    '''
    收集web数据
    '''

    def collect_v2(self):
        '''
        收集必应图片网站的图片
        :return:
        '''
        import requests  # 获取html的包
        from bs4 import BeautifulSoup  # 查找html特定元素
        import json

        # get html
        url = 'https://cn.bing.com/images/search?q=大侧脸'
        r = requests.get(url,
                         headers={'user-agent': 'Mozilla/5.0'})  # Mozilla代理

        # extract img_urls (提取方式和网站html格式有关)
        html_beautiful = BeautifulSoup(r.text, 'html.parser')  #
        temp = html_beautiful.find_all('a', class_='iusc')  #
        img_urls = list(
            map(lambda x: json.loads(x.attrs['m'])['murl'], temp))  # <a m='"murl":"img_url","purl"...,"turl":...'>

        # get imgs
        save_dir = '/home/xiancai/face_angle/6DRepNet/pics/'
        for ind, url in enumerate(img_urls):  #
            r = requests.get(url, headers={'user-agent': 'Mozilla/5.0'})
            # save
            save_path = f'{save_dir}{ind}.jpg'
            with open(save_path, 'wb') as f:
                f.write(r.content)
            print(f'{ind}:{len(img_urls)}:saved to {save_path}')


# def temp():
#     ls=glob.glob('/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/*')
#     for i in ls:
#         os.rename(i,i[:-3]+'jpg')

class Scene:
    '''
    处理现场收集的数据
    '''

    def temp(self):
        '''
        测试官方模型在大侧脸等上的效果
        :return:
        '''
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/*'
        save_dir = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/res_scene1/'
        import classify_pt
        inf = classify_pt.inference()

        ls = glob.glob(imgs_glob)
        for ind, i in enumerate(ls):
            p, y, r = map(int, inf.classify(i))
            p_ratio, y_ratio = max(1 - abs(p) / 90, 0.001), max(1 - abs(y) / 90, 0.001)
            q = round(2 * p_ratio * y_ratio / (p_ratio + y_ratio), 2)

            lab = f'{q},{p},{y},{r}'
            img = cv2.imread(i)
            img = cv2.resize(img, (224, 224))
            color = [0, 0, 255] if q < 0.6 else [0, 255, 0]
            cv2.putText(img, lab, (20, 20), 0, 0.5, color, thickness=2)

            save_path = save_dir + i.split('/')[-1]
            cv2.imwrite(save_path, img)
            print(f'{ind}/{len(ls)}')

    def temp2(self):
        '''
        抓图背景较大, 再crop
        :return:
        '''
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/origin/*'
        save_dir = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/origin_crop/'
        ls = glob.glob(imgs_glob)
        for ind, i in enumerate(ls):
            img = cv2.imread(i)
            h, w = img.shape[:2]  # padding 0.5
            # to padding 0.15
            x1 = int(w / 2 * 0.35)
            x2 = w - x1
            y1 = int(h / 2 * 0.35)
            y2 = h - y1
            img = img[y1:y2, x1:x2, :]
            # save
            save_path = save_dir + i.split('/')[-1]
            cv2.imwrite(save_path, img)
            print(f'{ind}/{len(ls)}')

    def clean(self):
        '''
        删除损坏文件
        :return:
        '''
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/origin/*'
        ls = glob.glob(imgs_glob)
        for i in ls:
            try:
                Image.open(i)
            except:
                print(f'{i} removed')
                os.remove(i)

    def add_lab(self):
        '''
        使用大模型对办公室数据打标
        :return:
        '''
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/origin/*'
        save_dir = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_02/origin_imgs_labs_a0/'
        import classify_pt
        inf = classify_pt.inference()
        ls = glob.glob(imgs_glob)
        for ind, i in enumerate(ls):
            p, y, r = map(int, inf.classify(i))
            # save
            save_path = save_dir + 'id'+i.split('/')[-1][:-4] +f'_{p}_{y}_{r}'+'.jpg'
            shutil.copy(i,save_path)
            print(f'{ind}/{len(ls)}')

    def make_txt(self):

        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/origin_imgs_labs/*'
        txt = '/data1/xiancai/FACE_ANGLE_DATA/2022_04_01/train2.txt'

        #
        ls = glob.glob(imgs_glob)
        # content = '\n'.join(map(lambda x: x + '  100  100  100', ls))re.split('_|-', imgname)
        content = '\n'.join(map(lambda x: x + '  '+'  '.join(re.split('_|.png',x.split('/')[-1])[1:4]), ls)) # ~/id1001_-7_-57_-6.jpg
        # save
        with open(txt,'w') as f:
            f.write(content)
        print(f'saved to {txt}')

    def make_testtxt(self):
        imgs_glob = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/*/*'
        txt = '/data1/xiancai/FACE_ANGLE_DATA/2022_03_31/scene1/test.txt'

        #
        ls = glob.glob(imgs_glob)
        content = '\n'.join(map(lambda x: x + ' '+ ('0' if x.split('/')[-2]=='0' else '1'), ls))
        # save
        with open(txt,'w') as f:
            f.write(content)
        print(f'saved to {txt}')

if  __name__ == '__main__':
    # Web_collection().collect_v2()
    # Pose_300W_LP().make_txt()
    # pass

    Scene().make_txt()

