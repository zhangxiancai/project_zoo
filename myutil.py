'''
性别识别：数据处理脚本
0: 男, 1: 女
'''
import glob
import os.path
import random

import cv2
import scipy.io as scio  # 读取mat文件
import numpy as np
import time


class util_time:
    '''
    统计程序的开始和结束时间
    '''
    sta_time=0
    fin_time=0

    def info_start_time(self):
        self.sta_time=time.time()
        sta_date = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        print(f'started at {sta_date}')

    def info_finish_time(self):
        self.fin_time=time.time()
        fin_date = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        print(f'finished at {fin_date}')

    def info_total_time(self):
        h=(self.fin_time-self.sta_time)/3600
        print(f'total time: {h}h')

class Base:
    '''
    make txt
    '''
    train_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_train.txt'
    val_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_val.txt'
    glob_str=''
    def get_lab_form_absolute_path(self,i):
        '''
        根据图片绝对路径名计算性别标签  0: 男, 1: 女
        :param i:
        :return: int
        '''
        pass

    def make_txt(self):
        '''
        制作识别数据集txt文件
        :return:
        '''

        train_txt = self.train_txt
        val_txt = self.val_txt
        if not os.path.exists(val_txt):
            os.mknod(val_txt)
        if not os.path.exists(train_txt):
            os.mknod(train_txt)
        # make strs
        ls = glob.glob(self.glob_str)
        strs = []
        for i in ls:
            if i[-3:] in ['jpg','png']:
                lab = self.get_lab_form_absolute_path(i)
                strs.append(i + ' ' + str(lab))  # [img lab]
            else:
                print(i)
        # divide train and val
        random.shuffle(strs)
        tag = int(0.9 * len(strs))
        # save
        with open(train_txt, 'w') as f:
            f.write('\n'.join(strs[:tag]))
        with open(val_txt, 'w') as f:
            f.write('\n'.join(strs[tag:]))

    def info(self):

        # read txt
        with open(self.train_txt,'r') as f:
            tra_ls=f.readlines()
        with open(self.val_txt, 'r') as f:
            val_ls = f.readlines()

        # count

        # # debug
        # for ind, i in enumerate(tra_ls):
        #     try:
        #         lab=int(i.strip().split(' ')[1])
        #     except:
        #         print(ind)
        #         print(i)
        # for ind, i in enumerate(val_ls):
        #     try:
        #         lab=int(i.strip().split(' ')[1])
        #     except:
        #         print(ind)
        #         print(i)

        tra_labs = list(map(lambda x: int(x.strip().split(' ')[1]), tra_ls))
        val_labs = list(map(lambda x: int(x.strip().split(' ')[1]), val_ls)) #[abso_path lab]

        tra_numb = len(tra_ls)
        tra_women_numb=sum(tra_labs)
        tra_men_numb = len(tra_labs)-tra_women_numb

        val_numb = len(val_ls)
        val_women_numb = sum(val_labs)
        val_men_numb = len(val_labs) - val_women_numb

        total=tra_numb+val_numb
        # total_women=tra_women_numb+val_men_numb # 错误
        total_women = tra_women_numb + val_women_numb
        total_men=tra_men_numb+val_men_numb

        # info
        print(self.__class__.__name__)
        print(f'train/men/women: {tra_numb} / {tra_men_numb} / {tra_women_numb}')
        print(f'val/men/women: {val_numb} / {val_men_numb} / {val_women_numb}')
        print(f'total/men/women: {total} / {total_men} / {total_women}\n')

        return tra_numb,val_numb,total,total_men,total_women

class AFAD(Base):
    '''
    AFAD数据集处理类
    '''
    train_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_train.txt'
    val_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_val.txt'
    glob_str='/data1/xiancai/FACE_DATA/AFAD-Full/*/*/*'

    def get_lab_form_absolute_path(self,i):
        lab = 0 if i.split('/')[-2] == '111' else 1
        return lab


class UTKface(Base):
    '''
    UTKface数据集处理类
    '''
    train_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_train.txt'
    val_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_val.txt'
    glob_str = '/data1/xiancai/FACE_DATA/UTKface/UTKFace/*'

    def get_lab_form_absolute_path(self,i):
        lab = int(i.split('/')[-1].split('_')[1])  # 性别 int
        return lab


class wiki(Base):
    '''
    wiki处理类
    数据从维基百科抓取, 错误较多，很多图片背景占比较大 2015
    '''

    def __init__(self):
        self.root_path = '/data1/xiancai/FACE_DATA/wiki/wiki_crop/'
        self.save_path = '/data1/xiancai/FACE_DATA/wiki/wiki_clean/'
        self.mat_path='/data1/xiancai/FACE_DATA/wiki/wiki_crop/wiki.mat'
        self.name='wiki'
        self.DEBUG=False

        self.train_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/wiki_train.txt'
        self.val_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/wiki_val.txt'
        self.glob_str = self.save_path+'*/*'

    def get_lab_form_absolute_path(self, i):
        lab=0 if i.split('/')[-1].split('_')[0]=='0' else 1
        return lab

    def clean(self):

        save_path = self.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mat = scio.loadmat(self.mat_path)  # mat格式参考https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
        file_paths = mat[self.name]['full_path'][0][0]  # 1*62328 ndarray
        gender = mat[self.name]['gender'][0][0]  # 1*62328 ndarray
        face_score = mat[self.name]['face_score'][0][0]
        second_face_score = mat[self.name]['second_face_score'][0][0]
        # face_location = mat[self.name]['face_location'][0][0]  # 自带的框不准，可能解码有问题，不使用此框

        root_path = self.root_path
        for ind, file_path in enumerate(file_paths[0]):
            # select 当图片有且只有一个人脸,且性别可确定
            if not face_score[0, ind] == float('-inf') and np.isnan(second_face_score[0, ind]) and not np.isnan(
                    gender[0, ind]):
                img = cv2.imread(root_path + file_path[0])

                # yolov5_face
                from detect_util.detect_yolov5_face_onnx import detect_one_onnx, torch
                det = detect_one_onnx(img)
                try:
                    x1, y1, x2, y2 = map(lambda x: int(x.item()), det[0, :4])
                    # 加padding
                    padding = int((y2-y1)*0.1) #
                    x1, y1, x2, y2 = x1-padding, y1-padding*2, x2+padding, y2+padding*2
                    # 扩充为正方形
                    w,h=x2-x1,y2-y1
                    s=abs(w-h)
                    if w<h:
                        x1,x2=x1-s//2,x2+s-s//2
                    else:
                        y1,y2=y1-s//2,y2+s-s//2
                    # crop
                    x1,y1=max(x1,0),max(y1,0)
                    x2,y2=min(x2,img.shape[1]),min(y2,img.shape[0])
                    img_crop = img[y1:y2,x1:x2,...]

                    # letterbox
                    h,w=img_crop.shape[:2]
                    new_s=max(w,h)
                    dw,dh=new_s-w,new_s-h
                    top, bottom, left, right=dh//2, dh-dh//2, dw//2, dw-dw//2
                    img_crop = cv2.copyMakeBorder(img_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

                except Exception as e:
                    print(f'cropped error: {file_path[0]}')
                    print(e.args)
                    continue

                # save
                sub_dir = save_path + file_path[0].split('/')[0] + '/'
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                lab=0 if gender[0, ind]==1 else 1 # 0:男性 1：女性
                file_name = str(lab) + '_' + file_path[0].split('/')[1]  # add gender lab to file_name {gender}_10049200_1891-09-16_1958.jpg
                try:
                    cv2.imwrite(sub_dir + file_name, img_crop)
                except:
                    print(f'saved error: {sub_dir}{file_name}')
                if self.DEBUG:
                    print(f'{ind}/{len(file_paths[0])}: saved to {sub_dir}{file_name} ')
        print('Done.')


class imdb(wiki):
    '''
    imdb处理类
    imdb数据的性别标签错误很多
    '''
    def __init__(self):
        # clean
        self.root_path = '/data1/xiancai/FACE_DATA/imdb/imdb_crop/'
        self.save_path = '/data1/xiancai/FACE_DATA/imdb/imdb_clean/'
        self.mat_path = '/data1/xiancai/FACE_DATA/imdb/imdb_crop/imdb.mat'
        self.name='imdb'
        self.DEBUG = False
        # make txt
        self.train_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/imdb_train.txt'
        self.val_txt = '/data1/xiancai/FACE_DATA/dataset_gender_classify/imdb_val.txt'
        self.glob_str = self.save_path+'*/*'

class kaggle_gender(Base):

    glob_str='/data1/xiancai/FACE_DATA/kaggle_gender/faces/*/*'
    train_txt='/data1/xiancai/FACE_DATA/dataset_gender_classify/kag_train.txt'
    val_txt='/data1/xiancai/FACE_DATA/dataset_gender_classify/kag_val.txt'

    def get_lab_form_absolute_path(self,i):

        lab=0 if i.split('/')[-2]=='man' else 1
        return lab

    def check(self):
        '''
        统计长宽不等的图片
        :return:
        '''
        ls=glob.glob(self.glob_str)
        tag=0
        for i in ls:
            img=cv2.imread(i)
            h,w=img.shape[:2]
            if h!=w: # 长宽不等
                print(f'{tag}: {i}')
                tag+=1


class seep(Base):
    glob_str='/data1/xiancai/FACE_DATA/seep/*/*'
    train_txt='/data1/xiancai/FACE_DATA/dataset_gender_classify/seep_train.txt'
    val_txt='/data1/xiancai/FACE_DATA/dataset_gender_classify/seep_val.txt'


    def get_lab_form_absolute_path(self, i):
        lab = 0 if i.split('/')[-2] == 'male' else 1
        return lab

class main:
    @staticmethod
    def info():
        '''
        统计所有数据集数量
        :return:
        '''
        ls=[AFAD(),kaggle_gender(),UTKface(),wiki(),imdb()]
        res=np.zeros((len(ls),5))
        for ind,i in enumerate(ls):
            res[ind,:]=i.info() #tra_numb,val_numb,total,total_men,total_women
        re=np.sum(res,axis=0)
        print(f'tra_numb,val_numb,total,total_men,total_women:{re}')




if __name__ == '__main__':
    ut=util_time()
    ut.info_start_time()

    # UTKface().make_txt()
    # kaggle_gender().check()
    # main().info()
    # seep().make_txt()
    main().info()


    ut.info_finish_time()
    ut.info_total_time()
