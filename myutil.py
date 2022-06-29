'''
数据处理工具

ccpd说明:https://blog.csdn.net/LuohenYJ/article/details/117752120
'''
import glob
import os
import json
import random
import shutil
import re
import time

import cv2
import numpy as np

class deepcam:
    '''
    处理神目数据
    xml:神目格式
    txt:yolo格式
    '''
    @staticmethod
    def make_jpg_xml():
        '''
        检测 step1
        :return:
        '''
        input_path = '/home/xiancai/DATA/PLATE_DATA/CCPD2021-IR/CCPD2021-IR/'
        jpg_path='/home/xiancai/DATA/PLATE_DATA/CCPD2021-IR/jpg/'
        xml_path='/home/xiancai/DATA/PLATE_DATA/CCPD2021-IR/xml/'
        if not os.path.exists(jpg_path):
            os.mkdir(jpg_path)
        if not os.path.exists(xml_path):
            os.mkdir(xml_path)

        ls=os.listdir(input_path)
        for ind,i in enumerate(ls):
            if i[-3:]=='jpg':
                shutil.copy(input_path+i,jpg_path+i)
            if i[-3:]=='xml':
                shutil.copy(input_path + i, xml_path + i)
            print(f'{ind}/{len(ls)}')
        print(f'total:{len(ls)}')
        print(f'jpg:{len(os.listdir(jpg_path))}')
        print(f'xml:{len(os.listdir(xml_path))}')

    @staticmethod
    def make_txt_from_xml():
        '''
        检测 step2: 将temp_xml数据转为temp_txt
        '''
        print(f'***************检测:step2*********************')
        tag_inf = dict()  # 标签分布

        # date = '11_27'
        # print(date)
        root='/home/xiancai/DATA/PLATE_DATA/CCPD2021-IR/'
        input_path = f'{root}xml/'  # json格式数据集路径
        txt_path = f'{root}txt/'  # txt（label）文件路径
        out_tag_inf =f'{root}txt_inf.txt'  # 标签分布信息存储地址

        if not os.path.exists(txt_path):
            os.mkdir(txt_path)

        ls = os.listdir(input_path)
        total = 0

        # 处理ls
        def single_jsontotxt(json_dir, out_dir):
            '''
            json转txt(车牌检测) 单个文件
            :param json_dir:输入json文件地址
            :param out_dir:输出txt文件地址  cls xywh landmark1234
            :return:
            '''
            # 读取 json 文件数据
            with open(json_dir, 'r') as load_f:
                content = json.load(load_f)
            # 循环处理
            filename = out_dir
            file_str = ''
            for t in content['faces']:
                # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值

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

                # landmark
                lu_x,lu_y = t['landmark1']['x']/W, t['landmark1']['y']/H
                ru_x,ru_y = t['landmark2']['x']/W, t['landmark2']['y']/H
                lb_x,lb_y = t['landmark3']['x']/W, t['landmark3']['y']/H
                rb_x,rb_y = t['landmark4']['x']/W, t['landmark4']['y']/H
                placeholder_x,placeholder_y=0,0 # 5对landmark 兼容yoloface
                # cat
                file_str += str(type) + ' ' \
                            + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + ' ' + str(round(h, 6)) + ' '\
                            + str(round(lu_x, 6)) + ' ' + str(round(lu_y, 6)) +' ' + str(round(ru_x, 6))+' ' + str(round(ru_y, 6)) + ' '\
                            + str(round(lb_x, 6))+' ' + str(round(lb_y, 6))+' ' + str(round(rb_x, 6))+' ' + str(round(rb_y, 6))+ ' '\
                            + str(round(placeholder_x, 6))+' ' + str(round(placeholder_y, 6))+'\n'

                # 记录数量
                if not tag_inf.get(t['type']):
                    tag_inf[t['type']] = 0
                tag_inf[t['type']] += 1

            # save
            if os.path.exists(filename):
                os.remove(filename)
            os.mknod(filename)  #
            fp = open(filename, mode="r+", encoding="utf-8")
            fp.write(file_str[:-1])
            fp.close()

        for i in ls:
            if i[-3:] == 'xml':
                out_fi = f'{txt_path}{i[:-7]}txt'
                single_jsontotxt(input_path + i, out_fi)  # 转换
                total += 1
                print(f'{total}/{int(len(ls))}: saved to {out_fi}')
        print(f'saved to {txt_path}')

        # save tag_inf
        if os.path.exists(out_tag_inf):
            os.remove(out_tag_inf)
        os.mknod(out_tag_inf)  #
        fp = open(out_tag_inf, mode="r+", encoding="utf-8")
        fp.write(str(tag_inf)+'\n'+str(sum(tag_inf.values())))
        fp.close()
        print(f'tag information: saved to {out_tag_inf}')
        print(tag_inf)
        print(sum(tag_inf.values()))

    @staticmethod
    def add_jpg_txt_to_plate_detect_dataset(input_img,input_txt,buff):
        '''
        检测 step3
        '''

        print(f'***************检测:step3*********************')
        # source

        # buff='CCPD2020_ccpd_green' # 加前缀
        # input_img=f'/home/xiancai/DATA/PLATE_DATA/{buff}/jpg/'
        # input_txt=f'/home/xiancai/DATA/PLATE_DATA/{buff}/txt/'
        print(f'input_img:{len(os.listdir(input_img))}')
        print((f'input_txt:{len(os.listdir(input_txt))}'))

        # dst
        dataset='/data1/xiancai/PLATE_DATA/plate_detect_dataset/'
        out_fir_tra_img=dataset+'images/train'
        out_fir_val_img = dataset+'images/val'

        out_fir_tra_txt=dataset+'labels/train'
        out_fir_val_txt = dataset+'labels/val'

        print('before:')
        t_i,v_i,t_tx,v_tx=len(os.listdir(out_fir_tra_img)),len(os.listdir(out_fir_val_img)),\
                          len(os.listdir(out_fir_tra_txt)),len(os.listdir(out_fir_val_txt))
        print(f'{out_fir_tra_img}:{t_i}')
        print(f'{out_fir_val_img}:{v_i}')
        print(f'{out_fir_tra_txt}:{t_tx}')
        print(f'{out_fir_val_txt}:{v_tx}')


        #划分 各类随机 抽两张/1%
        ls_img=os.listdir(input_img)
        # ls_txt=os.listdir(input_txt)
        random.shuffle(ls_img)
        # flag=3
        ls_img_tra=[]
        ls_img_val=[]

        flag=max(int(0.01*len(ls_img))+1,1) # 1%

        ls_img_val = ls_img[:flag]
        ls_img_tra = ls_img[flag:]
        # 添加至验证集
        for i in ls_img_val:
            shutil.copy(input_img+i,out_fir_val_img+f'/{buff}_{i}') #添加至val_img

            name_txt = input_txt + i[:-4] + '.txt' #
            if os.path.exists(name_txt):#如果有标签文件
                shutil.copy(name_txt,out_fir_val_txt+f'/{buff}_{i[:-4]}.txt') #添加至val_lab

        # 添加至训练集
        for ind,i in enumerate(ls_img_tra):
            shutil.copy(input_img + i, out_fir_tra_img+f'/{buff}_{i}')  # 添加至tra_img

            name_txt=input_txt+i[:-4]+'.txt'
            if os.path.exists(name_txt):
                shutil.copy(name_txt, out_fir_tra_txt+f'/{buff}_{i[:-4]}.txt')  # 添加至tra_lab
            print(f'{buff} {ind}/{len(ls_img_tra)}:')
            print('\ttrainset:'+out_fir_tra_img+f'/{buff}_{i}')
            print('\ttrainset:'+out_fir_val_txt + f'/{buff}_{i[:-4]}.txt')

        #check
        print('check:')
        t_i_d,v_i_d,t_tx_d,v_tx_d=len(os.listdir(out_fir_tra_img)),len(os.listdir(out_fir_val_img)),\
                          len(os.listdir(out_fir_tra_txt)),len(os.listdir(out_fir_val_txt))
        print(f'{out_fir_tra_img}:{t_i_d}')
        print(f'{out_fir_val_img}:{v_i_d}')
        print(f'img added: {t_i_d+v_i_d-t_i-v_i}')
        print(f'{out_fir_tra_txt}:{t_tx_d}')
        print(f'{out_fir_val_txt}:{v_tx_d}')
        print(f'lab added: {t_tx_d+v_tx_d-t_tx-v_tx}')

    @staticmethod
    def make_txt_from_xml_v2():
        '''
        jv_clean: xml (from labelme) to txt : cls xywh landmark1234
        '''

        xml_glob='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/*/*xml'
        ls = glob.glob(xml_glob)
        def xml2txt_one(xml_path, txt_path):
            '''
            json转txt(车牌检测) 单个文件
            :param xml_path:
            :param txt_path:输出txt文件地址  cls xywh landmark1234
            :return:
            '''
            # 读取 xml 文件数据
            with open(xml_path, 'r') as load_f:
                content = json.load(load_f)
            # 循环处理
            file_str = ''
            for t in content['faces']:
                # json越界修正
                x1, x2, y1, y2 = t['x'], t['x'] + t['w'], t['y'], t['y'] + t['h']
                H, W = content['image_height'], content['image_width']
                if H == '' or W =='':
                    H, W = cv2.imread(i[:-4]).shape[:2]
                x1, x2 = max(x1, 0), min(x2, W)
                y1, y2 = max(y1, 0), min(y2, H)
                if x1 > W or x2 < 0 or y1 > H or y2 < 0:
                    print(f'json label error: {xml_path} x1 {x1},y1 {y1},x2 {x2},y2 {y2}')
                    raise
                # convert to xywh
                x = (x1 + (x2 - x1) / 2) / W
                y = (y1 + (y2 - y1) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                # type
                type = 0  # 只检测
                # landmarks
                lu_x,lu_y = t['landmark1']['x']/W, t['landmark1']['y']/H
                ru_x,ru_y = t['landmark2']['x']/W, t['landmark2']['y']/H
                lb_x,lb_y = t['landmark3']['x']/W, t['landmark3']['y']/H
                rb_x,rb_y = t['landmark4']['x']/W, t['landmark4']['y']/H
                placeholder_x,placeholder_y=0,0 # 5对landmark 兼容yoloface
                # cat
                file_str += str(type) + ' ' \
                            + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + ' ' + str(round(h, 6)) + ' '\
                            + str(round(lu_x, 6)) + ' ' + str(round(lu_y, 6)) +' ' + str(round(ru_x, 6))+' ' + str(round(ru_y, 6)) + ' '\
                            + str(round(lb_x, 6))+' ' + str(round(lb_y, 6))+' ' + str(round(rb_x, 6))+' ' + str(round(rb_y, 6))+ ' '\
                            + str(round(placeholder_x, 6))+' ' + str(round(placeholder_y, 6))+'\n'

            # save
            if os.path.exists(txt_path):
                os.remove(txt_path)
            os.mknod(txt_path)  #
            fp = open(txt_path, mode="r+", encoding="utf-8")
            fp.write(file_str[:-1])
            fp.close()

        for ind,i in enumerate(ls):
            xml2txt_one(i, i[:-3]+'txt')  # 转换
            print(f'deepcam to yolo {ind}/{len(ls)}: saved to {i[:-3]}txt')

    @staticmethod
    def update_yolov5():
        '''
        jv_clean: make yolov5 or add a unit to yolov5
        '''

        # source
        # jpg='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/*/*g'
        jpg = '/data1/xiancai/PLATE_DATA/zhwei/jv_clean/*/*/*g'
        # txt='/data1/xiancai/PLATE_DATA/zhwei/jv_clean/*/*/*txt'
        buff=''

        # dst
        # yolov5='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow_yolov5/'
        yolov5='/data1/xiancai/PLATE_DATA/zhwei/jv_clean_yolov5/'
        out_fir_tra_img = yolov5 + 'images/train/'
        out_fir_val_img = yolov5 + 'images/val/'
        out_fir_tra_txt = yolov5 + 'labels/train/'
        out_fir_val_txt = yolov5 + 'labels/val/'
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
        ls_img = glob.glob(jpg)
        random.shuffle(ls_img)
        flag = max(int(0.01 * len(ls_img)) + 1, 1)  # 1%
        ls_img_val = ls_img[:flag]
        ls_img_tra = ls_img[flag:]

        # 添加至验证集
        for i in ls_img_val:
            # jpg
            nam = i.split('/')[-1]
            shutil.copy(i, out_fir_val_img + nam)  # 添加至val_img
            # lab
            txt_abpath = i+'.txt'  #
            if os.path.exists(txt_abpath):  # 如果有标签文件
                shutil.copy(txt_abpath, out_fir_val_txt + nam[:-3]+'txt')

        # 添加至训练集
        for ind, i in enumerate(ls_img_tra):
            # jpg
            nam = i.split('/')[-1]
            shutil.copy(i, out_fir_tra_img + nam)  # 添加至tra_img
            # lab
            txt_abpath = i + '.txt'
            if os.path.exists(txt_abpath):
                shutil.copy(txt_abpath, out_fir_tra_txt + nam[:-3] + 'txt')  # 添加至tra_lab

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

    @staticmethod
    def draw_txt():
        img_path='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/anhui/anhuiA00296.jpg'
        txt_path = '/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/anhui/anhuiA00296.jpg.txt'
        save_path = '/data1/xiancai/PLATE_DATA/other/test_04_27/txt.jpg'
        img = cv2.imread(img_path)
        with open(txt_path) as f:
            ls=f.readlines()
        H,W = img.shape[:2]
        for i in ls:
            its=list(map(float, i.strip().split(' ')))
            # draw box
            x, y, w, h = its[1:5]
            x1,x2,y1,y2 = int((x-w/2)*W), int((x+w/2)*W), int((y-h/2)*H), int((y+h/2)*H)
            img=cv2.rectangle(img,(x1,y1),(x2,y2),color=[0,255,0])
            # draw landmarks
            landmarks=its[5:]
            for i in range(5):
                lx,ly=landmarks[i*2:i*2+2]
                lx,ly=int(lx*W),int(ly*H)
                cv2.circle(img,(lx,ly),radius=i*2+1,color=[0,255,0])
        cv2.imwrite(save_path,img)

    @staticmethod
    def draw_xml():
        img_path='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/anhui/anhuiA00296.jpg'
        xml_path = '/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/anhui/anhuiA00296.jpg.xml'
        save_path ='/data1/xiancai/PLATE_DATA/other/test_04_27/xml.jpg'
        img = cv2.imread(img_path)
        with open(xml_path) as f:
            cont=json.load(f)

        for face in cont['faces']:
            # draw box
            x1,y1,w,h=int(face['x']),int(face['y']),int(face['w']),int(face['h'])
            x2,y2=x1+w,y1+h
            img=cv2.rectangle(img,(x1,y1),(x2,y2),color=[0,255,0])
            # draw landmarks
            for i in range(1,5):
                lx,ly=int(face[f'landmark{i}']['x']),int(face[f'landmark{i}']['y'])
                cv2.circle(img,(lx,ly),radius=i*2+1,color=[0,255,0])
        tag=cv2.imwrite(save_path,img)
        print(tag)

class ccpd:
    # 处理ccpd数据
    @staticmethod
    def ccpd_make_txt(tag=''):
        input_path=f'/home/xiancai/DATA/PLATE_DATA/CCPD2020_ccpd_green/origin/'
        txt_path = f'/home/xiancai/DATA/PLATE_DATA/CCPD2020_ccpd_green/txt/'
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)

        def convert_jpg_to_txt(jpg_name,txt_path):

            img=cv2.imread(input_path+jpg_name)
            W,H=img.shape[1],img.shape[0]
            x1y1x2y2,landmarks=jpg_name.split('-')[2:4] # 前提 一张图片一个车牌

            # xywh
            x1, y1, x2, y2 = map(int, re.split('_|&', x1y1x2y2))
            x = (x1 + (x2 - x1) / 2) / W
            y = (y1 + (y2 - y1) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H

            # landmarks
            landmarks_ls=list(map(int,re.split('_|&', landmarks)))
            lu_x, lu_y = landmarks_ls[4] / W, landmarks_ls[5]/ H
            ru_x, ru_y = landmarks_ls[6] / W, landmarks_ls[7] / H
            lb_x, lb_y = landmarks_ls[2] / W, landmarks_ls[3] / H
            rb_x, rb_y = landmarks_ls[0] / W, landmarks_ls[1] / H
            placeholder_x, placeholder_y = 0, 0  # 5对landmark 兼容yoloface

            # type
            type=0

            # cat
            file_str = str(type) + ' ' \
                        + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + ' ' + str(round(h, 6)) + ' ' \
                        + str(round(lu_x, 6)) + ' ' + str(round(lu_y, 6)) + ' ' + str(round(ru_x, 6)) + ' ' + str(
                round(ru_y, 6)) + ' ' \
                        + str(round(lb_x, 6)) + ' ' + str(round(lb_y, 6)) + ' ' + str(round(rb_x, 6)) + ' ' + str(
                round(rb_y, 6)) + ' ' \
                        + str(round(placeholder_x, 6)) + ' ' + str(round(placeholder_y, 6)) + '\n'
            # save
            if os.path.exists(txt_path):
                os.remove(txt_path)
            os.mknod(txt_path)  #
            fp = open(txt_path, mode="r+", encoding="utf-8")
            fp.write(file_str[:-1])
            fp.close()

        ls=os.listdir(input_path)
        total=0
        for i in ls:
            out_fi = f'{txt_path}{i[:-3]}txt'
            convert_jpg_to_txt( i, out_fi)  # 转换
            total += 1
            print(f'{tag} {total}/{int(len(ls))}: saved to {out_fi}')
        print(f'saved to {txt_path}')

    @staticmethod
    def ccpd_check_txt():

        txt_input ='/home/xiancai/DATA/PLATE_DATA/CCPD2019/txt/135301724138-90_84-0&398_691&606-698&624_2&618_1&393_697&399-0_0_33_26_15_33_32-158-201.txt'
        jpg_input ='/home/xiancai/DATA/PLATE_DATA/CCPD2019/CCPD2019/ccpd_base/135301724138-90_84-0&398_691&606-698&624_2&618_1&393_697&399-0_0_33_26_15_33_32-158-201.jpg'

        txt_input='/home/xiancai/DATA/PLATE_DATA/202101020-ok/txt/0009c5ca09087ecf0247855e3bc3a362.txt'
        jpg_input='/home/xiancai/DATA/PLATE_DATA/202101020-ok/jpg/0009c5ca09087ecf0247855e3bc3a362.jpg'
        with open(txt_input,'r') as f:
            its=f.readlines()[0].split(' ')
        img0=cv2.imread(jpg_input)

        # 画box
        H, W = img0.shape[:2]
        x, y, w, h = map(float,its[1:5])
        x1, y1, x2, y2 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
        pt1 = [x1, y1]
        pt2 = [x2, y2]
        cv2.rectangle(img0, pt1, pt2, (0, 255, 0), 2)

        for i in range(4):
            cen=(int(float(its[5+i*2])*W),int(float(its[5+i*2+1])*H))
            cv2.circle(img0, cen, radius=1, color=(255, 0, 0), thickness=20)
        tag=cv2.imwrite('/home/xiancai/DATA/PLATE_DATA/debug.jpg',img0)
        print(tag)

    @staticmethod
    def resize_plate_detect_dataset():
        input='/data1/xiancai/PLATE_DATA/plate_detect_dataset/'
        output='/data1/xiancai/PLATE_DATA/plate_detect_dataset_0.5ccpd/'
        img_tra='images/train/'
        img_val='images/val/'
        lab_tra='labels/train/'
        lab_val='labels/val/'

        orig_img_val=os.listdir(input+img_val)
        new_img_val=[]
        for i in orig_img_val:
            if i[:12]=='202101020-ok' or i[:11]=='CCPD2021-IR':
                new_img_val.append(i)
            elif random.random()<0.5:
                new_img_val.append(i)

        orig_img_tra=os.listdir(input+img_tra)
        new_img_tra=[]
        for i in orig_img_tra:
            if i[:12]=='202101020-ok' or i[:11]=='CCPD2021-IR': #如果为神目数据
                new_img_tra.append(i)
            elif random.random()<0.5:
                new_img_tra.append(i)

        print(len(new_img_val))
        print(len(new_img_tra))

        # val set
        for ind,i in enumerate(new_img_val):
            shutil.copy(input+img_val+i,output+img_val+i) #copy img
            txt_path=input+ lab_val+i[:-4]+'.txt'
            if os.path.exists(txt_path):
                shutil.copy(txt_path,output+lab_val+i[:-4]+'.txt') #copy lab
            print(f'{ind}/{len(new_img_val)}')

        # train
        for ind,i in enumerate(new_img_tra):
            shutil.copy(input+img_tra+i,output+img_tra+i) #copy img
            txt_path=input+ lab_tra+i[:-4]+'.txt'
            if os.path.exists(txt_path):
                shutil.copy(txt_path,output+lab_tra+i[:-4]+'.txt') #copy lab
            print(f'{ind}/{len(new_img_tra)}')

        # check
        print(len(os.listdir(output+img_tra)))
        print(len(os.listdir(output + img_val)))
        print(len(os.listdir(output + lab_tra)))
        print(len(os.listdir(output + lab_val)))

        print(len(new_img_val))
        print(len(new_img_tra))

    @staticmethod
    def resize_plate_detect_dataset_mutil_thread():
        input='/data1/xiancai/PLATE_DATA/plate_detect_dataset/'
        output='/data1/xiancai/PLATE_DATA/plate_detect_dataset_0.25ccpd/'
        img_tra='images/train/'
        img_val='images/val/'
        lab_tra='labels/train/'
        lab_val='labels/val/'

        orig_img_val=os.listdir(input+img_val)
        new_img_val=[]
        for i in orig_img_val:
            if i[:12]=='202101020-ok' or i[:11]=='CCPD2021-IR':
                new_img_val.append(i)
            elif random.random()<0.5:
                new_img_val.append(i)

        orig_img_tra=os.listdir(input+img_tra)
        new_img_tra=[]
        for i in orig_img_tra:
            if i[:12]=='202101020-ok' or i[:11]=='CCPD2021-IR': #如果为神目数据
                # new_img_tra.append(i)
                pass
            elif random.random()<0.0001:
                new_img_tra.append(i)

        print(len(new_img_val))
        print(len(new_img_tra))

        # # val set
        # for ind,i in enumerate(new_img_val):
        #     shutil.copy(input+img_val+i,output+img_val+i) #copy img
        #     txt_path=input+ lab_val+i[:-4]+'.txt'
        #     if os.path.exists(txt_path):
        #         shutil.copy(txt_path,output+lab_val+i[:-4]+'.txt') #copy lab
        #     print(f'{ind}/{len(new_img_val)}')

        import threading
        class myThread(threading.Thread):
            def __init__(self, threadID, data):
                threading.Thread.__init__(self)
                self.threadID = threadID
                # self.name = name
                # self.counter = counter
                self.data=data
            def run(self):
                # print("开始线程：" + self.threadID)
                sub_process(self.threadID, self.data)
                # print("退出线程：" + self.threadID)

        def sub_process(process_id,new_img_tra):
            # train set
            for ind,i in enumerate(new_img_tra):
                shutil.copy(input+img_tra+i,output+img_tra+i) #copy img
                txt_path=input+ lab_tra+i[:-4]+'.txt'
                if os.path.exists(txt_path):
                    shutil.copy(txt_path,output+lab_tra+i[:-4]+'.txt') #copy lab
                print(f'thread_id:{process_id} {ind}/{len(new_img_tra)}')

        total=len(new_img_tra)
        pro1=myThread(0,new_img_tra[:int(total*0.25)])
        pro2=myThread(1, new_img_tra[int(total*0.25):int(total*0.5)])
        pro3=myThread(2, new_img_tra[int(total*0.5):int(total*0.75)])
        pro4=myThread(3, new_img_tra[int(total*0.75):])
        pros=[pro1,pro2,pro3,pro4]
        for pro in pros:
            pro.start()
        for pro in pros:
            pro.join()

        # check
        print(len(os.listdir(output+img_tra)))
        print(len(os.listdir(output + img_val)))
        print(len(os.listdir(output + lab_tra)))
        print(len(os.listdir(output + lab_val)))

        print(len(new_img_val))
        print(len(new_img_tra))

class labelme:
    glob_str='/data1/xiancai/PLATE_DATA/zhwei/patch_v3.0_20220301/Yellow/*/*g'

    def labelme_to_deepcam_one(self,
                               labelme_json='/data1/xiancai/PLATE_DATA/zhwei/jv_clean/Black/beijing/beijingA00000.jpg.json',
                               deepcam_xml='/data1/xiancai/PLATE_DATA/zhwei/jv_clean/Black/beijing/beijingA00000.jpg.xml'):
        '''
        labelme格式 to 神目格式
        :param labelme_json:
        :param deepcam_xml:
        :return:
        '''
        # labelme_json='/data1/xiancai/PLATE_DATA/zhwei/jv_clean/Black/beijing/beijingA00000.jpg.json'
        # deepcam_xml='/data1/xiancai/PLATE_DATA/zhwei/jv_clean/Black/beijing/beijingA00000.jpg.xml'
        if not os.path.exists(labelme_json):
            return
        with open(labelme_json) as f:
            json_cont=json.load(f)
        if not os.path.exists(deepcam_xml):
            os.mknod(deepcam_xml)

        xml_dict={}
        xml_dict['filename']=''
        xml_dict['image_height']=''
        xml_dict['image_width']=''
        xml_dict['path']=''
        xml_dict['faces'] = []

        for it in json_cont['shapes']:
            face={}
            ps=it['points']
            face['landmark1'],face['landmark2'],face['landmark3'],face['landmark4']=\
                {'x':ps[0][0],'y': ps[0][1]},{'x':ps[1][0],'y': ps[1][1]},{'x':ps[3][0],'y': ps[3][1]},{'x':ps[2][0],'y': ps[2][1]}

            # x1,y1,x2,y2=ps[0][0],ps[0][1],ps[2][0],ps[2][1]
            ps = np.array(ps)
            x1, y1, x2, y2 = np.min(ps[:,0]).item(),np.min(ps[:,1]).item(),np.max(ps[:,0]).item(),np.max(ps[:,1]).item()
            face['h'], face['w'], face['x'], face['y'] =y2-y1,x2-x1,x1,y1
            face['type']=it['label']
            xml_dict['faces'].append(face)

        # save
        str=json.dumps(xml_dict,ensure_ascii=False)
        with open(deepcam_xml,'w',encoding='utf-8') as f:
            f.write(str)

    def labelme_to_deepcam_muti(self):
        ls = glob.glob(self.glob_str)
        for ind, i in enumerate(ls):
            labelme=i+'.json'
            deepcam=i+'.xml'
            self.labelme_to_deepcam_one(labelme,deepcam)
            print(f'labelme to deepcam {ind}/{len(ls)}: saved to {deepcam}')

from tqdm import tqdm
ls=[1,2,3,4,5]
ls=tqdm(ls)
for i in ls:
    time.sleep(1)
    # print(f'hello{i}')
    ls.set_description(f'{i}')

if __name__=='__main__':
    # # tag=['ccpd_blur','ccpd_db','ccpd_fn','ccpd_np','ccpd_rotate','ccpd_tilt','ccpd_weather']
    # tag = ['ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']
    # for i in tag:
    #     ccpd_make_txt(i)
    # jpgs_root='/home/xiancai/DATA/PLATE_DATA/CCPD2019/CCPD2019/'
    # jpgs=['ccpd_base/','ccpd_blur/','ccpd_challenge/','ccpd_db/','ccpd_fn/','ccpd_rotate/','ccpd_tilt/','ccpd_weather/','ccpd_np/']
    # txts_root='/home/xiancai/DATA/PLATE_DATA/CCPD2019/'
    # txts=['ccpd_base_txt/','ccpd_blur_txt/','ccpd_challenge_txt/','ccpd_db_txt/','ccpd_fn_txt/','ccpd_rotate_txt/','ccpd_tilt_txt/','ccpd_weather_txt/','ccpd_np_txt/']
    # buffs=['CCPD2019_ccpd_base','CCPD2019_ccpd_blur','CCPD2019_ccpd_challenge','CCPD2019_ccpd_db','CCPD2019_ccpd_fn','CCPD2019_ccpd_rotate','CCPD2019_ccpd_tilt','CCPD2019_ccpd_weather','CCPD2019_ccpd_np']
    # for jpg,txt,buff in zip(jpgs,txts,buffs):
    #     # print(jpg,txt,buff)
    #     add_jpg_txt_to_plate_detect_dataset(jpgs_root+jpg,txts_root+txt,buff)

    # labelme().labelme_to_deepcam_muti()
    # deepcam().make_txt_from_xml_v2()

    # deepcam().draw_xml()
    # deepcam().draw_txt()

    # deepcam().update_yolov5()
    pass

