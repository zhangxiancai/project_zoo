'''
车牌识别数据处理脚本
'''
import glob
import os
import random
import re
import shutil

import cv2
import numpy as np
import json

ccpd_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                  "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ccpd_ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
ccpd_ads = np.array(ccpd_ads)


def ccpd_preprocess(input_path, output_path, tag_adjust=True):
    '''
    扣图，resize，车牌号作为文件名后缀，save
    :return:
    '''
    # input_path='/data1/xiancai/PLATE_DATA/CCPD2020_ccpd_green/jpg/' # ccpd图片
    # output_path='/data1/xiancai/PLATE_DATA/plate_classify_dataset/CCPD2020_ccpd_green/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ls = os.listdir(input_path)
    for ind, i in enumerate(ls):
        img = cv2.imread(input_path + i)
        sub_names = i.split('-')
        x1y1x2y2, plate_num = sub_names[2], sub_names[-3]  # 前提 一张图片一个车牌
        x1, y1, x2, y2 = map(int, re.split('_|&', x1y1x2y2))
        plate_nums = list(map(int, plate_num.split('_')))
        # crop
        img = img[y1:y2, x1:x2, ...]

        # adjust
        if tag_adjust:
            rb_x, rb_y, lb_x, lb_y, lu_x, lu_y, ru_x, ru_y = map(int, re.split('_|&', sub_names[3]))
            points_src = np.float32([[lu_x - x1, lu_y - y1], [ru_x - x1, ru_y - y1], [lb_x - x1, lb_y - y1]])
            img = adjust(img, points_src)
        else:
            img = cv2.resize(img, (94, 24))
        # filename
        filename = i[:-4] + '-' + ccpd_provinces[plate_nums[0]] + ''.join(ccpd_ads[plate_nums[1:]]) + '.jpg'
        # save
        cv2.imwrite(output_path + filename, img)
        print(f'{ind}/{len(ls)}: saved to {output_path + filename}')


def debug_ccpd_preprocess():
    '''
    扣图，车牌号作为文件名后缀，处理单张ccpd图片
    :return:
    '''
    input_path = '/data1/xiancai/PLATE_DATA/other/'
    i = '011561302681992337-83_246-320&530_456&608-444&580_320&608_337&547_456&530-0_0_3_29_31_30_24_30-99-73.jpg'
    img = cv2.imread(input_path + i)
    sub_names = i.split('-')
    x1y1x2y2, plate_num = sub_names[2], sub_names[-3]  # 前提 一张图片一个车牌
    x1, y1, x2, y2 = map(int, re.split('_|&', x1y1x2y2))
    plate_nums = list(map(int, plate_num.split('_')))
    # crop
    img = img[y1:y2, x1:x2, ...]
    # filename
    filename = i[:-4] + '-' + ccpd_provinces[plate_nums[0]] + ''.join(ccpd_ads[plate_nums[1:]]) + '.jpg'
    # save
    cv2.imwrite(input_path + filename, img)
    print(f'saved to {input_path + filename}')

class deepcam:
    '''
    处理神目打标数据，扣出车牌，用于识别数据集
    '''
    origin_dir='/data1/xiancai/PLATE_DATA/yello_326/origin/'
    jpg_dir = origin_dir[:-7] + 'jpg/'
    xml_dir = origin_dir[:-7] + 'xml/'
    plate_dir='/data1/xiancai/PLATE_DATA/yello_326/plate/'

    def make_jpg_xml(self):

        if not os.path.exists(self.jpg_dir):
            os.mkdir(self.jpg_dir)
        if not os.path.exists(self.xml_dir):
            os.mkdir(self.xml_dir)
        ls=glob.glob(self.origin_dir+'*')
        for i in ls:
            if i[-3:]=='jpg':
                shutil.copy(i,self.jpg_dir)
            if i[-3:]=='xml':
                shutil.copy(i,self.xml_dir)

    def make_plate(self, tag_adjust=True):
        '''
        扣图，adjust，车牌号作为文件名后缀，save
        :return:
        '''
        jpg_path=self.jpg_dir
        xml_path=self.xml_dir
        plate_path=self.plate_dir

        if not os.path.exists(plate_path):
            os.mkdir(plate_path)
        xmls = os.listdir(xml_path)
        for ind, xml in enumerate(xmls):
            with open(xml_path + xml, 'r') as f:
                content = json.load(f)
            img_orig = cv2.imread(jpg_path + xml[:-4])
            for t in content['faces']:
                # filename
                filename = xml[:-8] + '_' + t['type'] + '.jpg'
                # print(f'{ind}/{len(xmls)}: saved to {plate_path + filename}')
                x1, y1, w, h = map(int, (t['x'], t['y'], t['w'], t['h']))
                # crop
                img = img_orig[y1:y1 + h, x1:x1 + w, ...]
                # adjust
                if tag_adjust:
                    lu_x, lu_y, ru_x, ru_y, lb_x, lb_y = int(t['landmark1']['x']), int(t['landmark1']['y']), int(
                        t['landmark2']['x']), int(t['landmark2']['y']), int(t['landmark3']['x']), int(t['landmark3']['y'])
                    points_src = np.float32([[lu_x - x1, lu_y - y1], [ru_x - x1, ru_y - y1], [lb_x - x1, lb_y - y1]])
                    img = adjust(img, points_src)
                else:
                    img = cv2.resize(img, (94, 24))

                cv2.imwrite(plate_path + filename, img)
                print(f'{ind}/{len(xmls)}: saved to {plate_path + filename}')

    def check_plate_name(self):
        '''
        clean 车牌识别数据（deepcam）
        :return:
        '''
        # training chars
        CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                 '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                 '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                 '新', '学', '警',
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                 ]
        CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

        ls = glob.glob(self.plate_dir+'*')
        err = []
        for ind, i in enumerate(ls):
            basename = os.path.basename(i)  # basename: '~_VE6J3B9.jpg','~-VE6J3B9.jpg'
            imgname, suffix = os.path.splitext(basename)
            imgname = re.split('_|-', imgname)[-1].strip()  #
            label = list()
            try:
                for c in imgname:
                    label.append(CHARS_DICT[c]) # 如果字符不在CHARS中（比如空格）
            except:
                err.append(i)
                print(f'error lab: {i}')
        for i in err:
            os.remove(i)
        print(err)


def make_train_val():
    '''
    制作train.txt val.txt
    :return:
    c'''
    # input
    imgs_path = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/*/*'
    # output
    train_txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/train.txt'
    val_txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
    if not os.path.exists(train_txt):
        os.mknod(train_txt)
    if not os.path.exists(val_txt):
        os.mknod(val_txt)
    ls = glob.glob(imgs_path)
    random.shuffle(ls)

    flag = int(0.9 * len(ls))
    # save
    with open(train_txt, 'w') as tr_f:
        tr_f.write('\n'.join(ls[:flag]))
    with open(val_txt, 'w') as val_f:
        val_f.write('\n'.join(ls[flag:]))

    print(flag)
    print(len(ls) - flag)
    print(f'Saved to {train_txt} and {val_txt}')


def check_plate_name():
    '''
    clean 车牌识别数据（deepcam）
    :return:
    '''
    # training chars
    CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新', '学', '警',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'I', 'O', '-'
             ]
    CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

    ls = glob.glob('/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/*/*')
    err = []
    for ind, i in enumerate(ls):
        basename = os.path.basename(i)  # basename: '~_VE6J3B9.jpg','~-VE6J3B9.jpg'
        imgname, suffix = os.path.splitext(basename)
        imgname = re.split('_|-', imgname)[-1].strip()  #
        label = list()
        try:
            for c in imgname:
                label.append(CHARS_DICT[c])
        except:
            err.append(i)
            print(i)
        print(f'{ind}/{len(ls)}')
    for i in err:
        os.remove(i)
    print(err)


def plate_adjust():
    '''
    车牌倾斜矫正
    :return:
    '''
    # input_path='/data1/xiancai/PLATE_DATA/other/011561302681992337-83_246-320&530_456&608-444&580_320&608_337&547_456&530-0_0_3_29_31_30_24_30-99-73-皖AD57606.jpg'
    input_path = '/data1/xiancai/PLATE_DATA/other/011561302681992337-83_246-320&530_456&608-444&580_320&608_337&547_456&530-0_0_3_29_31_30_24_30-99-73.jpg'
    out_path = '/data1/xiancai/PLATE_DATA/debug/'
    img_orig = cv2.imread(input_path)
    # x1y1x2y2 and landmark
    name = input_path.split('/')[-1]
    sub_names = name.split('-')
    x1y1x2y2, landmarks = sub_names[2], sub_names[3]
    x1, y1, x2, y2 = map(int, re.split('_|&', x1y1x2y2))
    rb_x, rb_y, lb_x, lb_y, lu_x, lu_y, ru_x, ru_y = map(int, re.split('_|&', landmarks))
    # crop
    img = img_orig[y1:y2, x1:x2, ...]
    cv2.imwrite(out_path + 'adebug_0.jpg', img)
    cv2.imwrite(out_path + 'adebug_reference.jpg', cv2.resize(img, (94, 24)))
    # adjust
    points_src = np.float32([[lu_x - x1, lu_y - y1], [ru_x - x1, ru_y - y1], [lb_x - x1, lb_y - y1]])
    # points_dst=np.float32([[0,0],[ru_x-lu_x,0],[0,lb_y-lu_y]])
    # size=(ru_x-lu_x,lb_y-lu_y)
    points_dst = np.float32([[0, 0], [94, 0], [0, 24]])
    size = (94, 24)
    M = cv2.getAffineTransform(points_src, points_dst)
    img = cv2.warpAffine(img, M, size)
    cv2.imwrite(out_path + 'adebug_1.jpg', img)
    # resize
    img = cv2.resize(img, (94, 24))
    cv2.imwrite(out_path + 'adebug_2.jpg', img)


def adjust(img, points_src):
    '''
    矫正
    :param img: cropped img (box)
    :param points_src=np.float32([[lu_x-x1,lu_y-y1],[ru_x-x1,ru_y-y1],[lb_x-x1,lb_y-y1]])
    :return: adjusted img 24*94
    '''
    points_dst = np.float32([[0, 0], [94, 0], [0, 24]])
    M = cv2.getAffineTransform(points_src, points_dst)
    img = cv2.warpAffine(img, M, (94, 24))
    return img


def make_plate_classify_dataset_adjust():
    output = f'/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/'
    # inputs = ['/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_base/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_blur/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_challenge/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_db/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_fn/',
    #           # '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_np/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_rotate/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_tilt/',
    #           '/data1/xiancai/PLATE_DATA/CCPD2019/CCPD2019/ccpd_weather/'
    #           ]
    inputs = ['/data1/xiancai/PLATE_DATA/CCPD2020_ccpd_green/jpg/']
    for i in inputs:
        sub_dir = i.split('/')[-3] + '_' + i.split('/')[-2] + '/'
        output_path = output + sub_dir
        ccpd_preprocess(input_path=i, output_path=output_path)
        # print(output_path)

    # inputs_deepcam = ['/data1/xiancai/PLATE_DATA/202101020-ok/jpg/',
    #                   '/data1/xiancai/PLATE_DATA/CCPD2021-IR/jpg/']
    # for i in inputs_deepcam:
    #     xml_path = i[:-4] + 'xml/'
    #     sub_dir = i.split('/')[-3] + '/'
    #     output_path = output + sub_dir
    #     deepcam_preprocess(jpg_path=i, xml_path=xml_path, plate_path=output_path)


class base:
    @staticmethod
    def read_filenames_from_txt(file_path):
        '''
        读取txt
        :param path: train.txt
        :return: [img_path] * ~
        '''
        with open(file_path, 'r') as f:
            ls = list(map(lambda x: x.strip(), f.readlines()))
        return ls


class info:
    '''
    统计车牌识别数据集信息
    '''
    names = ['blue', 'green', 'yellow']
    KEMANS_K=2
    train_txt='/data1/xiancai/PLATE_DATA/plate_classify_dataset/train.txt'
    val_txt='/data1/xiancai/PLATE_DATA/plate_classify_dataset/val.txt'

    def get_region(self, img_path):
        '''
        车牌分割
        :param img_path:
        :return:
        '''
        img = cv2.imread(img_path)
        # img = img[..., ::-1]  # to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # to hsv
        # img[...,1:]=0 to h
        # calculate hm  rgb值的分布
        hm = {}

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not hm.get(tuple(img[i, j, :])):
                    hm[tuple(img[i, j, :])] = [1, [[i, j]]]  # { rgb: [count,ijs] }
                else:
                    hm[tuple(img[i, j, :])][0] += 1
                    hm[tuple(img[i, j, :])][1].append([i, j])
        # print(hm)
        # sorted(hm.values())

        # calculate labs (kmeans)
        keys = np.array(list(hm.keys()))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, labs, ps = cv2.kmeans(keys.astype(np.float32), self.KEMANS_K, bestLabels=None, criteria=criteria, attempts=1,
                                 flags=cv2.KMEANS_PP_CENTERS)  # https://blog.csdn.net/lanshi00/article/details/104109963
        keys = list(map(tuple, list(keys)))  # rgb值
        labs = list(map(int, list(labs)))  # rgb值的区域编号
        return keys, labs, hm, ps

    def show_region(self, keys, labs, hm, ps,save_path='/data1/xiancai/PLATE_DATA/other/debug/region.jpg'):
        '''
        展示分割后的图片
        :param keys: rgbs
        :param labs:
        :param hm: { rgb: [count,ijs] }
        :param ps:
        :return:
        '''
        keys = list(map(tuple, list(keys)))
        labs = list(map(int, list(labs)))
        img_debug = np.zeros((24, 94, 3))
        # cls_map={0:(0),1:(255),2:(127)}
        ps = ps.astype(np.uint8)
        for ind, cls in enumerate(labs):
            ijs = hm[keys[ind]][1]
            for ij in ijs:
                img_debug[ij[0], ij[1], :] = ps[cls, :]
                # img_debug[ij[0], ij[1], :] = color[color_cls[cls]]

        cv2.imwrite(save_path, img_debug[..., ::-1])

    def init_color_reference_set(self):
        '''
        获得三种车牌的RGB值
        :return: 字典refer_set: color:[背景rgbs,字体rgbs]
        '''

        img_paths = ['/data1/xiancai/PLATE_DATA/img_reference/blue_1.jpg',
                     '/data1/xiancai/PLATE_DATA/img_reference/green_1.jpg',
                     '/data1/xiancai/PLATE_DATA/img_reference/yellow_1.jpg']
        refer_set = {}

        for index, img_path in enumerate(img_paths):
            keys, labs, _, _ = self.get_region(img_path)  # labs从0开始

            # count labs and get the sorted region
            hm = {} # lab:count
            for i in labs:
                if not hm.get(i):
                    hm[i] = 1
                else:
                    hm[i] += 1
            hm_sorted = dict(sorted(hm.items(), key=lambda x: x[1], reverse=True))  # 将hm按照v降序排列
            cls_sorted = list(hm_sorted.keys())  #

            # max_v = 0
            # for k, v in hm.items():
            #     if v > max_v:
            #         max_k = k
            #         max_v = v

            # save blue_refer
            back_refer = []
            font_refer=[]
            for ind, i in enumerate(labs): # labs: [cls_id], keys:[rgb]
                if i == cls_sorted[0]:
                    back_refer.append(keys[ind])
                if i == cls_sorted[1]:
                    font_refer.append(keys[ind])
            refer_set[self.names[index]] = [back_refer,font_refer]  # rgb值
        return refer_set

    @staticmethod
    def count_yellow_plate():
        txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
        img_paths = base.read_filenames_from_txt(txt)
        # img_paths=['/data1/xiancai/PLATE_DATA/other/debug/0210668103448-88_92-316&386_534&490-538&469_328&494_328&407_538&382-0_0_0_27_25_24_26-30-35-皖AA3102.jpg']
        random.shuffle(img_paths)
        img_paths = img_paths[:100]
        for index, i in enumerate(img_paths):
            print(str(index) + ': ' + i)
            img = cv2.imread(i)
            img = img[..., ::-1]  # to RGB

            # calculate hm  rgb值的分布
            hm = {}

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if not hm.get(tuple(img[i, j, :])):
                        hm[tuple(img[i, j, :])] = [1, [[i, j]]]  # [count,ijs]
                    else:
                        hm[tuple(img[i, j, :])][0] += 1
                        hm[tuple(img[i, j, :])][1].append([i, j])
            # print(hm)
            # sorted(hm.values())

            # calculate labs (kmeans)
            keys = np.array(list(hm.keys()))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, labs, ps = cv2.kmeans(keys.astype(np.float32), 3, bestLabels=None, criteria=criteria, attempts=1,
                                     flags=cv2.KMEANS_PP_CENTERS)  # https://blog.csdn.net/lanshi00/article/details/104109963

            # count color
            ps = ps.astype(np.uint8)
            color_cls = []
            #
            # yellow,blue,green,black=(225,225,0),(0,225,225),(0,255,0),(0,0,0)
            color = np.array(((225, 225, 0), (0, 0, 225), (0, 255, 255), (0, 0, 0)))
            for p in range(3):
                col = np.argmin(np.sqrt(np.sum((ps[None, p] - color) ** 2, axis=1)))  # 特征向量所属于的颜色
                color_cls.append(col)
            print(color_cls)
            print(ps)

            # debug: show labs
            keys = list(map(tuple, list(keys)))
            labs = list(map(int, list(labs)))
            img_debug = np.zeros((24, 94, 3))
            # cls_map={0:(0),1:(255),2:(127)}
            ps = ps.astype(np.uint8)
            for ind, cls in enumerate(labs):
                ijs = hm[keys[ind]][1]
                for ij in ijs:
                    img_debug[ij[0], ij[1], :] = ps[cls, :]
                    # img_debug[ij[0], ij[1], :] = color[color_cls[cls]]
            cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/kmeans/kmean_{index}.jpg', img_debug[..., ::-1])
            cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/kmeans/img_{index}.jpg', img[..., ::-1])
            s = 0

    def count_yellow_plate_v2(self):
        txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
        img_paths = base.read_filenames_from_txt(txt)
        refer_set = self.init_color_reference_set()  # 初始化三种车牌的参考RGB值
        # random.shuffle(img_paths)
        img_paths = img_paths[:100]
        for index, i in enumerate(img_paths):
            print(str(index) + ': ' + i)
            img = cv2.imread(i)
            img = img[..., ::-1]  # to RGB
            # 统计待识别车牌在refer_set上的分布
            color_count_dict = {}
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    #
                    for k, rgbs in refer_set.items():
                        if tuple(img[i, j, :]) in rgbs:  # rgb是否属于该种车牌
                            if not color_count_dict.get(k):
                                color_count_dict[k] = 1
                            else:
                                color_count_dict[k] += 1
            # plate name
            max_v = 0
            max_k = ''
            for k, v in color_count_dict.items():
                if v > max_v:
                    max_k = k
                    max_v = v
            print(refer_set.keys())
            print(color_count_dict)
            print(max_k)

    def classify_color(self,img_path = '/home/xiancai/DATA/TEST/img_reference/green_1.jpg'):
        # img_path = '/home/xiancai/DATA/TEST/img_reference/green_1.jpg'
        # refer_set = self.init_color_reference_set()  # 初始化三种车牌的参考RGB值
        # template=[(120, 255, 255),(60, 255, 225),(30, 255, 225)] # 蓝，绿，黄，hsv
        # template= {'blue':120,'green':60,'yellow':30}
        template = {'blue': 230, 'green': 120, 'yellow': 60}
        img = cv2.imread(img_path)
        # img = img[..., ::-1]  # to RGB
        # 统计待识别车牌在refer_set上的分布
        color_count_dict = {}
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                #
                for k, h in template.items():
                    if img[i, j, :][0] < h+30 and img[i, j, :][0] > h-30:  # rgb是否属于该种车牌
                        if not color_count_dict.get(k):
                            color_count_dict[k] = 1
                        else:
                            color_count_dict[k] += 1
        # plate name
        max_v = 0
        max_k = ''
        for k, v in color_count_dict.items():
            if v > max_v:
                max_k = k
                max_v = v
        print(template.keys())
        print(color_count_dict)
        print(max_k)


    def count_chinese_char(self):
        '''
        依据txt文件统计中文字符分布
        :return:
        '''
        ls=base.read_filenames_from_txt(self.train_txt)
        hm = {}
        for i in ls:
            plate=re.split('_|-',i.split('/')[-1])[-1].strip() # 皖AJ654Y.jpg 注意要strip
            char=plate[0]

            # count
            if not hm.get(char):
                hm[char]=1
            else:
                hm[char]+=1

        hm_sorted = dict(sorted(hm.items(), key=lambda x: x[1], reverse=True))  # 将hm按照v降序排列
        hm_str='\n'.join(map(lambda x:x[0]+': '+str(x[1]),hm_sorted.items()))
        print(hm_str)
        print(hm_sorted)

    def test_kmeans(self):
        '''
        测试车牌分割效果
        :return:
        '''
        txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
        img_paths = base.read_filenames_from_txt(txt)
        # random.shuffle(img_paths)
        img_paths = img_paths[:100]
        for index, i in enumerate(img_paths):
            if i.split('/')[-2] == 'CCPD2019_ccpd_base':
                print(str(index) + ': ' + i)
                keys, labs, hm, ps = self.get_region(i)
                self.show_region(keys, labs, hm, ps,f'/data1/xiancai/PLATE_DATA/other/debug/kmeans2/kmeans_{index}.jpg')
                cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/kmeans2/img_{index}.jpg', cv2.imread(i))

    def test_change_color(self):
        '''
        测试改变颜色 #效果不行
        :return:
        '''
        txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
        img_paths = base.read_filenames_from_txt(txt)
        # random.shuffle(img_paths)
        img_paths = img_paths[:100]
        for index, i in enumerate(img_paths):
            if i.split('/')[-2] == 'CCPD2019_ccpd_base':
                print(str(index) + ': ' + i)
                self.classify_color(i)



class augment(info):
    '''
    自定义车牌数据增强
    '''
    def __init__(self):
        super(augment, self).__init__()
        self.yellow_refer = self.init_color_reference_set()['yellow']  # [背景rgbs,字体rgbs]

    def select_rgb_from_yell_refer(self):
        '''
        选择背景rgb,字体rgb
        :return:
        '''
        p_back=random.randint(0,len(self.yellow_refer[0])-1)
        p_font = random.randint(0, len(self.yellow_refer[1])-1)
        # bf_rgbs = [(225, 225, 0), (0, 0, 0), None]  # 背景颜色，字体颜色，_
        template=[self.yellow_refer[0][p_back],self.yellow_refer[1][p_font],None]
        return template



    def change_plate_color(self, img_path='/data1/xiancai/PLATE_DATA/img_reference/blue_1.jpg'):
        '''
        改变车牌颜色
        :param img_path:
        :return:
        '''

        # get region
        keys, labs, hm, ps = self.get_region(img_path)  # labs从0开始
        self.show_region(keys, labs, hm, ps)
        # count labs and get the sorted region
        hm_lab = {}  # { cls_id: count }
        for i in labs:
            if not hm_lab.get(i):
                hm_lab[i] = 1
            else:
                hm_lab[i] += 1
        hm_lab_sorted = dict(sorted(hm_lab.items(), key=lambda x: x[1], reverse=True))  # 将hm_lab按照v降序排列
        cls_sorted = list(hm_lab_sorted.keys())  #

        # template 选择一次
        template = self.select_rgb_from_yell_refer()  # 背景颜色，字体颜色，_
        # change color
        img_origin = cv2.imread(img_path)
        img_origin = img_origin[..., ::-1]  # to rgb
        img_new = img_origin.copy()
        keys = list(map(tuple, list(keys)))
        labs = list(map(int, list(labs)))
        for ind, cls in enumerate(labs):
            ijs = hm[keys[ind]][1]
            for ij in ijs:
                # # template
                # template = self.select_rgb_from_yell_refer()  # 背景颜色，字体颜色，_
                if template[cls_sorted.index(cls)]:
                    img_new[ij[0], ij[1], :] = template[cls_sorted.index(cls)] # change
        # cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/change_color.jpg', img_new[..., ::-1])
        return img_new[..., ::-1] # bgr


    def test_change_plate_color(self):
        '''
        测试
        :return:
        '''
        txt = '/data1/xiancai/PLATE_DATA/plate_classify_dataset_adjust/val.txt'
        img_paths = base.read_filenames_from_txt(txt)
        random.shuffle(img_paths)
        img_paths = img_paths[:100]
        for index, i in enumerate(img_paths):
            print(str(index) + ': ' + i)
            img_new=self.change_plate_color(i)
            cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/augment/{index}_change.jpg', img_new)
            cv2.imwrite(f'/data1/xiancai/PLATE_DATA/other/debug/augment/{index}_aimg.jpg', cv2.imread(i))


if __name__ == '__main__':
    # deepcam().make_jpg_xml()
    augment().test_change_color()