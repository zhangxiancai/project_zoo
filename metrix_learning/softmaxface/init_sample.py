'''
注册图片
'''

import random
import shutil
import os

def init_sample():
    '''
    从分类数据集input_dir中选择各类别图片一张 注册到output_dir
    :return:
    '''
    input_dir='/home/xiancai/DATA/FIRE_DATA/fire_12_03/temp_classify/'
    output_dir='/home/xiancai/fire-equipment-demo/classify_metric_learning/register_images_12_03/'
    ls = os.listdir(input_dir)
    hm=dict() #
    res=[]

    # 各类别数据分别保存在hm
    for i in ls:
        if i in ('test.txt','train.txt'):
            continue
        ty=i.split('_')[0]
        hm[ty]=[i] if not hm.get(ty) else hm[ty]+[i]

    dates=['10_15','11_23','11_24','11_27','11_28','11_30','12_01','12_03']
    dates.reverse()

    for v in hm.values():

        # per type
        hm_temp = dict()
        # 按日期保存至hm_temp
        for i in v:
            d=i.split('_d')[1][:5]
            hm_temp[d]=[i] if not hm_temp.get(d) else hm_temp[d]+[i]

        for da in dates: # 优先选择最近日期的
            if hm_temp.get(da):
                random.shuffle(hm_temp[da])
                res+=hm_temp[da][:5] # 选择n个
                break

    # save
    for i,r in enumerate(res):
        shutil.copy(input_dir+r,output_dir+r)
        print(f'{i}: saved to {output_dir+r}')

init_sample()