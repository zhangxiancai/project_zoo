'''
数据增强:生成黄牌
by zxc, refer leijd
'''
import sys
import os
sys.path.append('/home/xiancai/plate/LPRNet_Pytorch/chinese_license_plate_generator')

from generate_multi_plate import MultiPlateGenerator
import plate_number
from plate_number import provinces,digits,letters # 车牌生成 支持的中文字符
import random
import cv2
import numpy as np

# 数据集中文字符分布
hm={'皖': 304695, '苏': 34393, '浙': 3592, '沪': 1863, '豫': 1738, '粤': 1267, '鄂': 1170, '鲁': 1146, '京': 1033, '赣': 800, '闽': 791, '湘': 689, '冀': 652, '陕': 578, '川': 428, '晋': 417, '渝': 405, '津': 373, '辽': 322, ' ': 286, '云': 143, '黑': 129, '甘': 120, '贵': 60, '蒙': 44, '桂': 42, '新': 32, '吉': 26, 'V': 17, '青': 15, '琼': 12, '宁': 9, '藏': 1, 'T': 1}

def select_province():

    char=provinces[np.random.randint(len(provinces))]
    if hm[char]>1000:
        char = provinces[np.random.randint(len(provinces))]
    if hm[char]>2000:
        char = provinces[np.random.randint(len(provinces))]

    return char

def generate_yellow_plate():
    '''
    单层黄牌 例：'皖A123B5' 或 '皖B123B学'
    :return:
    '''

    # generator = MultiPlateGenerator('chinese_license_plate_generator/plate_model', 'chinese_license_plate_generator/font_model')
    generator = MultiPlateGenerator('/home/xiancai/plate/LPRNet_Pytorch/chinese_license_plate_generator/plate_model/',
                                    '/home/xiancai/plate/LPRNet_Pytorch/chinese_license_plate_generator/font_model/')
    # plate numb
    if random.random()<0.9:
        plat_numb=plate_number.generate_plate_number_blue()
    else:
        plat_numb = plate_number.generate_plate_number_yellow_xue()
    plate=select_province()+plat_numb[1:]
    type = 'green_truck' if random.random()<0.5 else 'yellow'
    img = generator.generate_plate_special(plat_numb, type, False)

    img=cv2.resize(img,(94,24))
    cv2.imwrite('debug_y.jpg',img)
    # print(f'generator plate: saved to debug_y.jpg')
    return img,plat_numb

if __name__=='__main__':
    generate_yellow_plate()