'''
数据处理工具
'''
import glob
import os
import random
import shutil
import cv2
import argparse
import json

def check_fal():
    '''查找训练集的负样本图片 （根据负样本无txt,正样本有txt）'''
    ls1=os.listdir('/home/xiancai/915_data_yolov5_s/images/train')
    ls2=os.listdir('/home/xiancai/915_data_yolov5_s/labels/train')

    for j in ls1:
        tmp=j[:-3] + 'txt'
        if tmp not in ls2:#image has no label
            print('no label: ',j)

def check_val():
    '''检查验证集分布'''
    ls_lab=os.listdir('/home/xiancai/Ruler/ruler/labels/val')
    ls_ima=os.listdir('/home/xiancai/Ruler/ruler/images/val')
    res=dict()
    ls_lab.sort()
    for i in ls_lab:
        with open('/home/xiancai/Ruler/ruler/labels/val'+'/'+i) as f:
            strs=f.readlines()
            for t in strs:
                t=t.split(' ')
                if not res.get(t[0]):
                    res[t[0]]=0
                res[t[0]]+=1
    print('图片数：', len(ls_ima))
    print('labels:', len(ls_lab))

    print('各类别数: ',res )
    print('objs: ',sum(res.values()))
    '''
    图片数： 128
    labels: 128
    各类别数:  {'0': 30, '3': 4, '7': 3, '10': 12, '11': 28, '6': 27, '2': 6, '4': 14, '5': 13, '8': 9, '9': 28, '1': 17}
    objs:  191
'''

def addFalseSamples(input_add,out_add):
    '''添加val集负样本（全黑图片）'''

    # input_add='/home/xiancai/no_ruler_0916/'
    # out_add='/home/xiancai/Ruler/ruler/images/val/'
    soruce = os.listdir(input_add)
    soruce=soruce[:10]#
    print('pre: val图片数',len(os.listdir(out_add)))
    for i in soruce:
        shutil.copy(input_add+i,out_add+'passblack_'+i)
    print('aft: val图片数 ',len(os.listdir(out_add)))


def addFalseSamples_tr(input_add,out_add):
    '''添加训练集负样本（全黑图片）'''


    soruce = os.listdir(input_add)
    soruce=soruce[10:]#

    print('pre: train图片数',len(os.listdir(out_add)))
    for i in soruce:
        if os.path.isfile(input_add+i):
            shutil.copy(input_add+i,out_add+'passblack_'+i)
    print('aft: train图片数 ',len(os.listdir(out_add)))


def copyAtoB(A='/home/xiancai/test/ima/*',B='/home/xiancai/test/ima2/',buf = 'buf1_'):
    '''
    将A（目录+模式）复制到B（目录），并给文件名加前缀buf1
    :param A: '/home/xiancai/test/ima/*'
    :param B: '/home/xiancai/test/ima2/'
    :param buf: 给所复制文件添加的前缀
    :return:
    '''

    print('Copy ', A, ' to ', B,':')
    ls=glob.glob(A)
    if not os.path.exists(B):
        os.makedirs(B)

    print('A:',len(ls),'B:',len(os.listdir(B)))
    for i in ls:
        b=i.split('/')
        b=B+buf+b[-1]
        shutil.copy(i,b)
        print('copied ',i,' to ',b)
    print('A:', len(glob.glob(A)), 'B:', len(os.listdir(B)))


def moveAtoB(A='/home/xiancai/test/ima/*',B='/home/xiancai/test/ima2/',buf = 'buf1_'):
    '''
    将A（目录+模式）剪切到B（目录），并给文件名加前缀buf1
    :param A: '/home/xiancai/test/ima/*'
    :param B: '/home/xiancai/test/ima2/'
    :param buf: 给所剪切文件添加的前缀
    :return:
    '''

    print('Move ',A,' to ',B,':')
    ls=glob.glob(A)
    if not os.path.exists(B):
        os.makedirs(B)
    print('A:', len(ls), 'B:', len(os.listdir(B)))
    for i in ls:
        b=i.split('/')
        b=B+buf+b[-1]
        if os.path.isfile(i):
            shutil.move(i,b)#
            print('moved ',i,' to ',b)
    print('A:', len(glob.glob(A)), 'B:', len(os.listdir(B)))


def divide_jsonimage(json_data_dir):
    '''
    将xml和jpg放入两个文件夹 dir+'/json/'  dir + '/image/'
    :param dir:
    :return:
    '''
    ls = os.listdir(json_data_dir)
    ls.sort()
    out_dir_label=json_data_dir+'json/'
    out_dir_image = json_data_dir + 'image/'
    if not os.path.exists(out_dir_label):
        os.makedirs(out_dir_label)
    if not os.path.exists(out_dir_image):
        os.makedirs(out_dir_image)
    t1, t2 = 0, 0
    for it in ls:
        if it[-3:] == 'xml':
            shutil.copy(json_data_dir + it, out_dir_label) #
            t1 += 1
        if it[-3:] == 'jpg':
            shutil.copy(json_data_dir + it, out_dir_image)
            t2 += 1

    print(t1,t2)
    imas=os.listdir(out_dir_image)
    labs=os.listdir(out_dir_label)
    for i in imas:
        if i+'.xml' not in labs:
            print('image:'+i+' 无标签')

def counts_files():
    input="/home/xiancai/firecontrol_data_1/1/"
    ls=os.listdir(input)
    res=0
    for i in ls:
        its=os.listdir(input+i)
        res+=len(its)
    print(res)#2425*2+24*2=4898

def divide_val():
    #划分消防数据验证集
    images_val='/home/xiancai/Ruler/Pytorch/firecontrol/images/val/'
    labels_val = '/home/xiancai/Ruler/Pytorch/firecontrol/labels/val/'

    images_tra='/home/xiancai/Ruler/Pytorch/firecontrol/images/train/'
    labels_tra='/home/xiancai/Ruler/Pytorch/firecontrol/labels/train/'
    if not os.path.exists(images_val):
        os.makedirs(images_val)
    if not os.path.exists(labels_val):
        os.makedirs(labels_val)

    ls=os.listdir('/home/xiancai/Ruler/Pytorch/firecontrol/images/train')
    random.shuffle(ls)
    ls=ls[:100]#
    for i in ls:
        shutil.copy(images_tra+i,images_val+i)
        shutil.copy(labels_tra+i[:-3]+'txt',labels_val+i[:-3]+'txt')
    print('images_val',len(os.listdir(images_val)))
    print('labels_val', len(os.listdir(labels_val)))
    # N=2425
    # ran=random.random(1,N)

def divide_ruler():
    #将新数据划分给ruler训练集和测试集
    output_1='/home/xiancai/Ruler/ruler/images/val/'
    output_2='/home/xiancai/Ruler/ruler/images/train/'
    # output_1='/home/xiancai/test/val/'
    # output_2='/home/xiancai/test/train/'
    input='/home/xiancai/ruler_before13_data/'

    if not os.path.exists(output_1):
        os.makedirs(output_1)
    if not os.path.exists(output_2):
        os.makedirs(output_2)
    ls=os.listdir(input)
    random.shuffle(ls)
    ls1=ls[:200]#
    ls2=ls[200:]

    print(output_1,len(os.listdir(output_1)))
    for i in ls1:
        shutil.copy(input+i,output_1+i)
        # shutil.copy(labels_tra+i[:-3]+'txt',labels_val+i[:-3]+'txt')
    print(output_1, len(os.listdir(output_1)))
    print()
    print(output_2, len(os.listdir(output_2)))
    for i in ls2:
        shutil.copy(input+i,output_2+i)
    print(output_2, len(os.listdir(output_2)))
    # print('labels_val', len(os.listdir(labels_val)))
    # N=2425
    # ran=random.random(1,N)

def read_anchors():
    import torch
    from models.experimental import attempt_load

    model = attempt_load('/home/xiancai/Ruler/Pytorch/runs/train/exp22/weights/best.pt', map_location=torch.device('cpu'))
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    print(m.anchor_grid)


def check_image():
    #检测图片是否完整
    import cv2
    input='/home/xiancai/ruler_before13_data/'
    ls=os.listdir(input)
    for i in ls:
        adr=input + i
        img=cv2.imread(adr)
        if img is None:
            print('error img:',i)
            os.remove(adr)
            print('img deleted:', i)

def renameFile():
    input='/home/xiancai/ruler_before13_data/'
    ls = os.listdir(input)
    for i in ls:
        os.rename(input+i,input+'pass_before13_'+i)
        print(input+'pass_before13_'+i)






def process_fire_11_23():
    '''
    检测 step1
    11月23日打标消防数据（文件夹结构：目录-子目录-数据）  将js,img 放入两个文件夹 加前缀
    {'6_灭火防护手套': 20, '2_多功能消防水枪': 19, '13_三分水器': 22, '10_抢险救援服': 16, '5_灭火防护服': 8,
     '17_消防栓扳手': 16, '23_止水器': 11, '22_直流水枪': 9, '12_抢险救援靴': 14}
    :return:
    '''
    # date='11_24'
    print(f'***************检测:step1*********************')
    print(date)
    root=f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/'
    input=root+'origin/'
    inputs=[]
    sub_dirs=os.listdir(input)
    for i in sub_dirs:
        inputs.append(input+i) #

    output_xml=root+'temp_xml/'
    output_jpg = root+'temp_jpg/'
    if not os.path.exists(output_xml):
        os.mkdir(output_xml)
    if not os.path.exists(output_jpg):
        os.mkdir(output_jpg)

    total=0
    id=0
    for inp in inputs:
        ls=os.listdir(inp)
        type = inp.split('/')[-1]#sub_dir为类别名
        total+=len(ls)
        for i in ls:
            if i[-3:]=='jpg':
                shutil.copy(inp+'/'+i,output_jpg+f'{type}_d{date}_{i}') # 加类别+日期前缀,放入一个文件夹
                pass
            if i[-3:]=='xml':
                # check xml
                with open(inp+'/'+i) as f:
                    cont=json.load(f)
                tys=[con['type'].split('_')[0] for con in cont['faces']]
                if type not in tys:
                    print(f'xml err: sub_dir of origin/ is different from types in xml {inp}/{i}')
                    raise

                shutil.copy(inp+'/'+i,output_xml+f'{type}_d{date}_{i}') # 加类别+日期前缀
                pass
        id += 1
        print(f'source:({inp}):{total}')
        print(f'out:{len(os.listdir(output_jpg))+len(os.listdir(output_xml))}')

    print(f'xml:{len(os.listdir(output_xml))}')
    print(f'jpg:{len(os.listdir(output_jpg))}')
    print(f'saved to {output_jpg} and {output_xml}')


def check_yolotojson():
    '''
    convert yolo to json
    用于检查 yolo 是否正确
    '''

    input='/home/xiancai/DATA/FIRE_DATA/fire_11_27/temp_txt/'
    out='/home/xiancai/DATA/FIRE_DATA/fire_11_27/temp_txt_xml_check/'
    if not os.path.exists(out):
        os.mkdir(out)

    H,W=960,1280 #图片高宽
    ls=os.listdir(input)
    total=0
    for i in ls:

        with open(input+i,'r') as f:
            strs=f.readlines()

        # convert yolo to json
        hm = dict()
        hm['faces'] = []
        for st in strs:
            s = list(map(float, st.split(' ')))  # xywh yolo
            face = dict()

            # j_type=int(s[0])
            # j_type = cls_names[int(s[0])] #
            j_type='object'
            j_w = int(s[3] * W)
            j_h = int(s[4] * H)
            j_x = int(s[1] * W - j_w / 2)
            j_y = int(s[2] * H - j_h / 2)
            face['type'], face['x'], face['y'], face['w'], face['h'] = j_type, j_x, j_y, j_w, j_h
            hm['faces'].append(face)
            # shutil.copy(input + i, output + 'in_pri_' + i)
        hm['filename'] = i
        hm['image_height'] = H
        hm['image_width'] = W
        hm['path'] = ''

        # save json
        fi_str = json.dumps(hm)
        fi_name = out + i[:-4] + '.jpg.xml' # 注意
        if os.path.exists(fi_name):
            os.remove(fi_name)
        os.mknod(fi_name)
        fp = open(fi_name, mode="r+", encoding="utf-8")
        fp.write(fi_str)
        fp.close()
        total += 1
        print(f'total:{total}')


def add_fire_11_23_to_firecontrol():
    '''
    检测 step3
    划分11月23日数据(txt+jpg), 添加至消防数据集
    '''

    print(f'***************检测:step3*********************')
    # source
    # date='11_24'
    print(f'date:{date}')
    root=f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/'
    input_img=root+'temp_jpg/'
    input_txt=root+'temp_txt/'
    print(f'input_img:{len(os.listdir(input_img))}')
    print((f'input_txt:{len(os.listdir(input_txt))}'))

    # dst
    dataset='/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/'
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
    hm_img=dict() #k:类别 v:文件名列表
    for i in ls_img:
        ty=i.split('_')[0]
        if not hm_img.get(ty):
            hm_img[ty]=[i]
        else:
            hm_img[ty].append(i)
    for k in hm_img.keys():
        random.shuffle(hm_img[k])
        flag=max(int(0.01*len(hm_img[k]))+1,1) # 1%
        # if len(hm_img[k])<5: #如果该类别数据量小于5
        #     flag=0
        # elif len(hm_img[k])<10:
        #     flag=1
        ls_img_val+=hm_img[k][:flag]
        ls_img_tra+=hm_img[k][flag:]


    # 添加至验证集 各2张
    for i in ls_img_val:
        shutil.copy(input_img+i,out_fir_val_img) #添加至val_img

        name_txt = input_txt + i[:-4] + '.txt' #
        if os.path.exists(name_txt):#如果有标签文件
            shutil.copy(name_txt,out_fir_val_txt) #添加至val_lab
    # 添加至训练集
    for i in ls_img_tra:
        shutil.copy(input_img + i, out_fir_tra_img)  # 添加至tra_img

        name_txt=input_txt+i[:-4]+'.txt'
        if os.path.exists(name_txt):
            shutil.copy(name_txt, out_fir_tra_txt)  # 添加至tra_lab
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


def remove_xxdate_from_fire_detect_dataset():
    # 删除检测数据集里的特定日期数据

    dataset='/home/xiancai/DATA/FIRE_DATA/fire_detect_dataset/'
    out_fir_tra_img=dataset+'images/train/'
    out_fir_val_img = dataset+'images/val/'

    out_fir_tra_txt=dataset+'labels/train/'
    out_fir_val_txt = dataset+'labels/val/'
    print('before:')
    t_i,v_i,t_tx,v_tx=len(os.listdir(out_fir_tra_img)),len(os.listdir(out_fir_val_img)),\
                      len(os.listdir(out_fir_tra_txt)),len(os.listdir(out_fir_val_txt))
    print(f'{out_fir_tra_img}:{t_i}')
    print(f'{out_fir_val_img}:{v_i}')
    print(f'{out_fir_tra_txt}:{t_tx}')
    print(f'{out_fir_val_txt}:{v_tx}')

    imgs=[out_fir_tra_img,out_fir_val_img]
    txts=[out_fir_tra_txt,out_fir_val_txt]

    # rm

    for t in (0,1):
        ls_img = os.listdir(imgs[t])
        for i in ls_img:
            da=i.split('_d')[1][:5]
            if da==date:
                os.remove(imgs[t]+i)
                print(imgs[t]+i)

        ls_txt=os.listdir(txts[t])
        for i in ls_txt:
            da=i.split('_d')[1][:5]
            if da == date:
                os.remove(txts[t]+i)
                print(txts[t]+i)

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

def set_classification_tra_val_fire_11_23():
    '''
    # 划分消防识别训练验证集 数据截止11月23日 resnet
    :return: train.txt 和 test.txt
    '''

    input='/home/xiancai/DATA/fire_11_23_2_classification/*/*'
    output_tra='/home/xiancai/DATA/fire_11_23_2_classification/train.txt'
    output_val='/home/xiancai/DATA/fire_11_23_2_classification/test.txt'
    if not os.path.exists(output_tra):
        os.mknod(output_tra)
    if not os.path.exists(output_val):
        os.mknod(output_val)

    ls=glob.glob(input)
    res=[]
    for i in ls:
        # img=cv2.imread(i)
        # if img is None:
        #     continue
        name=i.split('/')
        x=name[-2]+'/'+name[-1]#图片地址
        # y=int(name[-1].split('_')[0])-1#类别
        y=int(name[-2])-1 # 补丁 使用子目录名-1 作为类别标签 注意
        res.append(x+' '+str(y))
    random.shuffle(res)
    size_tra=int(len(res)*0.9)

    #训练集
    with open(output_tra, 'w') as f:
        for i in range(0,size_tra):
            f.write(res[i] + '\n')
    #验证集
    with open(output_val, 'w') as f:
        for i in range(size_tra,len(res)):
            f.write(res[i] + '\n')
    print(f'saved to {output_tra} and {output_val}')


def changeImages():
    #识别 step1
    #抠图 制作识别数据集

    # date='11_23_2'
    print(f'***************识别:step1*********************')
    print(f'date: {date}')
    root=f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/'

    input_img=root+'temp_jpg/'
    # input_lab=root+'temp_txt/'
    input_xml=root+'temp_xml/'
    out_file_dir=root+'temp_classify/'
    if not os.path.exists(out_file_dir):
        os.mkdir(out_file_dir)


    ls_img=os.listdir(input_img)
    # ls_lab=os.listdir(input_lab)
    # ls_img=['5_WIN_20201229_10_07_53_Pro.jpg']#debug
    tag_1=0#succeed
    tag_2=0#non-square
    fals=[]#fail
    total_11_23=0
    for i in ls_img:
        json_dir = input_xml + i + '.xml'
        if not os.path.exists(json_dir):# 如果没有标签
            print(f'img {i} has no xml')
            continue
        # 读图片
        img=cv2.imread(input_img+i)

        # 读标签
        # with open(input_lab+i[:-3]+'txt') as f:
        #     lines=f.readlines()


        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)

        for ind,t in enumerate(content['faces']):
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

        # for line in lines[:1]:
        #     ts=list(map(float,line.split(' ')))
            H,W=img.shape[0],img.shape[1]

            #check yolo标签
            x,y,w,h=int(x*W),int(y*H),int(w*W),int(h*H)
            c1, c2 = max(x - int(w/2),0), min(x + int(w/2),W)
            r1, r2 = max(y - int(h/2),0), min(y + int(h/2),H)#yolo标签越界修正（11/2425）
            x,y,w,h=(c1+c2)//2,(r1+r2)//2,c2-c1,r2-r1

            # 转为正方形
            s=int(max(w,h)/2)
            c1,c2=x-s,x+s
            r1,r2=y-s,y+s

            #越界填充
            if c1<0:
                pad=-c1
                img=cv2.copyMakeBorder(img,0,0,pad,0,cv2.BORDER_CONSTANT,0)
                c1,c2=c1+pad,c2+pad
            if r1 < 0:
                pad = -r1
                img=cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
                r1, r2 = r1 + pad, r2 + pad
            if c2 > img.shape[1]:
                pad=c2-img.shape[1]
                img=cv2.copyMakeBorder(img,0,0,0,pad,cv2.BORDER_CONSTANT,0)
            if r2 > img.shape[0]:
                pad=r2-img.shape[0]
                img=cv2.copyMakeBorder(img,0,pad,0,0,cv2.BORDER_CONSTANT,0)

            # cv2.imwrite('/home/xiancai/test/'+i, img)#debug

            # #越界撤销
            # if r1<0 or r2>H or c1<0 or c2>W:
            #     tag_2+=1
            #     c1, c2 = max(x - int(w/2),0), min(x + int(w/2),W)
            #     r1, r2 = max(y - int(h/2),0), min(y + int(h/2),H)#yolo标签越界修正（11/2425）

            #越界舍弃
            # if r1<0:
            #     r1,r2=0,r2+r1
            # if c1<0:
            #     c1,c2=0,c2+c1
            # if r2>H:
            #     r2,r1=H,r1+r2-H
            # if c2>W:
            #     c2,c1=W,c1+c2-W

            res=img[r1:r2,c1:c2,...]#截取

            # tys=i.split('_')
            # if len(tys)==2: # 补丁 如果为11_23数据则文件前缀加1 之前版本的图片文件名从1开始
            #     tys[0] = str(int(tys[0])+1)
            #     print(f'11_23 total:{total_11_23}')
            #     total_11_23+=1


            # save
            type=int(t['type'].split('_')[0]) #xml '0_65水带'
            if date=='10_15':
                type-=1 # 补丁：兼容10_15 xml
            name=f'{type}_{ind}_cls_{i}'
            try:
                cv2.imwrite(out_file_dir+name,res)
                tag_1 += 1
            except:
                fals.append(name)
            finally:
                # log
                print(out_file_dir + name, '(total succeed fail non-square):',len(ls_img),tag_1,len(fals),tag_2)


    for i in fals:
        print(i,'false')
    print(f'temp_classify:{len(os.listdir(out_file_dir))}')
    print(f'saved to {out_file_dir}')

def add_classify_dataset():
    # 识别 step2
    # date='11_23'
    print(f'***************识别:step2*********************')
    root=f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/'
    input=root+'temp_classify/'
    output='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    ls=os.listdir(input)
    t_0=len(os.listdir(output))
    print(f'temp_classify:{len(ls)}')
    print(f'before fire_classify_dataset:{t_0}')
    for i in ls:
        shutil.copy(input+i,output+i)
    t_1=len(os.listdir(output))
    print('check:')
    print(f'after fire_classify_dataset:{t_1} added:{t_1-t_0}')

def set_classify_tra_val():
    # 识别 step3
    # 划分消费识别训练验证集
    #配置消防识别数据集标签 resnet

    # input='/home/xiancai/firecontrol_class_data_2/*/*'
    # output_tra='/home/xiancai/Ruler/pytorch-image-classfication/data/train.txt'
    # output_val='/home/xiancai/Ruler/pytorch-image-classfication/data/test.txt'
    # if not os.path.exists(output_tra):
    #     os.mknod(output_tra)
    # if not os.path.exists(output_val):
    #     os.mknod(output_val)
    #
    # ls=glob.glob(input)
    # res=[]
    # for i in ls:
    #     # img=cv2.imread(i)
    #     # if img is None:
    #     #     continue
    #     name=i.split('/')
    #     x=name[-2]+'/'+name[-1]#图片地址
    #     y=int(name[-1].split('_')[0])-1#类别
    #     res.append(x+' '+str(y))
    # random.shuffle(res)
    # size_tra=int(len(res)*0.9)
    #
    # with open(output_tra, 'w') as f:
    #     for i in range(0,size_tra):
    #         f.write(res[i] + '\n')
    # with open(output_val, 'w') as f:
    #     for i in range(size_tra,len(res)):
    #         f.write(res[i] + '\n')

    print(f'***************识别:step3*********************')
    input='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    output_tra=input+'train.txt'
    output_val=input+'test.txt'

    if not os.path.exists(output_tra):
        os.mknod(output_tra)
    if not os.path.exists(output_val):
        os.mknod(output_val)

    ls=os.listdir(input)
    res=[]
    for i in ls:
        if i=='train.txt' or i== 'test.txt':
            continue
        x=i #图片地址
        y=i.split('_')[0] #类别
        res.append(x+' '+y)

    #划分 各类随机抽10%
    # flag=2
    ls_img_tra=[]
    ls_img_val=[]
    hm_img=dict() #k:类别 v:文件名列表
    for i in res:
        ty=i.split(' ')[-1] # 取前缀为类别
        if not hm_img.get(ty):
            hm_img[ty]=[i]
        else:
            hm_img[ty].append(i)
    for k in hm_img.keys():
        random.shuffle(hm_img[k])
        flag=max(int(0.1*len(hm_img[k])),1)# 各类别10%
        ls_img_val+=hm_img[k][:flag]
        ls_img_tra+=hm_img[k][flag:]

    random.shuffle(ls_img_val)
    random.shuffle(ls_img_tra) # 打乱顺序，一个batch内的数据类别更加丰富 重要！！！
    with open(output_tra, 'w') as f:
        for i in ls_img_tra:
            f.write(i + '\n')
    with open(output_val, 'w') as f:
        for i in ls_img_val:
            f.write(i + '\n')

    print(f'total:{len(ls)-2}')
    print(f'saved to {output_tra} :{len(ls_img_tra)} and {output_val}:{len(ls_img_val)}')


def classify_test():
    #放入子目录 划分
    input='/home/xiancai/DATA/FIRE_DATA/fire_10_15/temp_classify/'
    output='/home/xiancai/DATA/FIRE_DATA/fire_10_15/test_clas/'
    ls=os.listdir(input)
    for i in ls:
        ty=i.split('_')[0]
        sub_dir=output+ty+'/'
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        shutil.copy(input+i,sub_dir+i)


    input='/home/xiancai/DATA/FIRE_DATA/fire_10_15/test_clas/*/*'
    output_tra='/home/xiancai/DATA/FIRE_DATA/fire_10_15/test_clas/train.txt'
    output_val='/home/xiancai/DATA/FIRE_DATA/fire_10_15/test_clas/test.txt'
    if not os.path.exists(output_tra):
        os.mknod(output_tra)
    if not os.path.exists(output_val):
        os.mknod(output_val)

    ls=glob.glob(input)
    res=[]
    for i in ls:
        # img=cv2.imread(i)
        # if img is None:
        #     continue
        name=i.split('/')
        x=name[-2]+'/'+name[-1]#图片地址
        y=int(name[-1].split('_')[0])#类别
        res.append(x+' '+str(y))

    random.shuffle(res)
    size_tra=int(len(res)*0.9)
    with open(output_tra, 'w') as f:
        for i in range(0,size_tra):
            f.write(res[i] + '\n')
    with open(output_val, 'w') as f:
        for i in range(size_tra,len(res)):
            f.write(res[i] + '\n')

    # print(f'total:{len(ls) - 2}')
    # print(f'saved to {output_tra} :{len(ls_img_tra)} and {output_val}:{len(ls_img_val)}')

def changeImages_test():
    #抠图

    input_img='/home/xiancai/DATA/FIRE_DATA/fire_10_15/temp_jpg/'
    input_lab='/home/xiancai/DATA/FIRE_DATA/fire_10_15/temp_txt/'
    out_img='/home/xiancai/DATA/FIRE_DATA/fire_10_15/test_class_origin/'
    if not os.path.exists(out_img):
        os.makedirs(out_img)

    ls_img=os.listdir(input_img)
    # ls_lab=os.listdir(input_lab)
    # ls_img=['5_WIN_20201229_10_07_53_Pro.jpg']#debug
    tag_1=0#succeed
    tag_2=0#non-square
    fals=[]#fail
    for i in ls_img:
        img=cv2.imread(input_img+i)#读图片
        with open(input_lab+i[:-3]+'txt') as f:
            line=f.read()
            ts=list(map(float,line.split(' ')))#读标签
        H,W=img.shape[0],img.shape[1]

        #check yolo标签
        x,y,w,h=int(ts[1]*W),int(ts[2]*H),int(ts[3]*W),int(ts[4]*H)
        c1, c2 = max(x - int(w/2),0), min(x + int(w/2),W)
        r1, r2 = max(y - int(h/2),0), min(y + int(h/2),H)#yolo标签越界修正（11/2425）
        x,y,w,h=(c1+c2)//2,(r1+r2)//2,c2-c1,r2-r1

        # 转为正方形
        s=int(max(w,h)/2)
        c1,c2=x-s,x+s
        r1,r2=y-s,y+s

        #越界填充
        if c1<0:
            pad=-c1
            img=cv2.copyMakeBorder(img,0,0,pad,0,cv2.BORDER_CONSTANT,0)
            c1,c2=c1+pad,c2+pad
        if r1 < 0:
            pad = -r1
            img=cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
            r1, r2 = r1 + pad, r2 + pad
        if c2 > img.shape[1]:
            pad=c2-img.shape[1]
            img=cv2.copyMakeBorder(img,0,0,0,pad,cv2.BORDER_CONSTANT,0)
        if r2 > img.shape[0]:
            pad=r2-img.shape[0]
            img=cv2.copyMakeBorder(img,0,pad,0,0,cv2.BORDER_CONSTANT,0)

        # cv2.imwrite('/home/xiancai/test/'+i, img)#debug

        # #越界撤销
        # if r1<0 or r2>H or c1<0 or c2>W:
        #     tag_2+=1
        #     c1, c2 = max(x - int(w/2),0), min(x + int(w/2),W)
        #     r1, r2 = max(y - int(h/2),0), min(y + int(h/2),H)#yolo标签越界修正（11/2425）

        #越界舍弃
        # if r1<0:
        #     r1,r2=0,r2+r1
        # if c1<0:
        #     c1,c2=0,c2+c1
        # if r2>H:
        #     r2,r1=H,r1+r2-H
        # if c2>W:
        #     c2,c1=W,c1+c2-W

        res=img[r1:r2,c1:c2,...]#截取

        tys=i.split('_')
        out_file_dir=out_img+tys[0]+'/'
        if not os.path.exists(out_file_dir):
            os.mkdir(out_file_dir)

        try:
            cv2.imwrite(out_file_dir+i,res)#保存
            tag_1 += 1
        except:
            fals.append(i)
        finally:
            # log
            print(out_file_dir + i, '(total succeed fail non-square):',len(ls_img),tag_1,len(fals),tag_2)
    for i in fals:
        print(i,'false')

def check_fire_classify_dataset():
    print('check fire_classify_dataset:')
    input='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    ls=os.listdir(input)
    for ind,i in enumerate(ls):
        if i=='test.txt' or i == 'train.txt':
            continue
        xml,dirc=i.split('_')[0],i.split('_')[3]
        if xml!=dirc:
            print(f'{ind}:{i}')
    print(f'total:{len(ls)}')


def get_txt_from_xml():
    '''
    检测 step2: 将temp_xml数据转为temp_txt
    '''
    print(f'***************检测:step2*********************')
    # cls_names = {0: '65水带', 1: '80水带', 2: '多功能消防水枪', 3: '黄色消防头盔', 4: '空呼气瓶9L', 5: '灭火防护服', 6: '灭火防护手套', 7: '灭火防护靴',
    #              8: '灭火防护腰带', 9: '泡沫枪PQ6',
    #              10: '抢险救援服', 11: '抢险救援头盔', 12: '抢险救援靴', 13: '三分水器', 14: '水域救援头盔', 15: '水域救援靴', 16: '水域漂浮背心',
    #              17: '消防栓扳手', 18: '新款黄色消防头盔', 19: '液压剪',
    #              20: '液压救援顶杆', 21: '正压式呼吸面罩', 22: '直流水枪', 23: '止水器', 24: '80公转80公转接头', 25: 'PQ8泡沫枪', 26: 'PQ16泡沫枪',
    #              27: '防高温手套', 28: '隔热服',
    #              29: '灭火防护头套', 30: '轻型防化服', 31: '消防员二类吊带', 32: '消防员接触式送受话器', 33: '消防员三类吊带', 34: '便携式移动消防灯组',
    #              35: '阀门堵漏套具', 36: '非接触式红外测温仪', 37: '隔离警示带',
    #              38: '呼吸器背架', 39: '混凝土切割锯', 40: '捆绑式堵漏气垫', 41: '炮口', 42: '无齿锯', 43: '消防专用救生衣', 44: '移动消防水炮',
    #              45: '6.8L正压式呼吸器', 46: '大力剪', 47: '地下消火栓扳手', 48: '二分水器',
    #              49: '红色消防头盔(全盔)', 50: '黄色消防头盔(全盔)', 51: '撬棍', 52: '铁锹', 53: '消防锤', 54: '消防尖斧', 55: '消防平斧', 56: '滤水器',
    #              57: 'f'}  # yolo:json
    tag_inf = dict()  # 标签分布

    # date = '11_27'
    # print(date)
    input = f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/temp_xml/'  # json格式数据集路径
    output = f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/temp_txt/'  # txt（label）文件路径
    if not os.path.exists(output):
        os.mkdir(output)

    ls = os.listdir(input)
    total = 0

    # 处理ls
    def single_jsontotxt(json_dir, out_dir):
        '''
        json转txt 单个文件
        :param json_dir:输入json文件地址
        :param out_dir:输出txt文件地址
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
            file_str += str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
                round(w, 6)) + ' ' + str(round(h, 6)) + '\n'

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
            out_fi = f'{output}{i[:-7]}txt'
            single_jsontotxt(input + i, out_fi)  # 转换
            total += 1
            print(f'{total}/{int(len(ls))}: saved to {out_fi}')
    print(f'saved to {output}')
    # save tag_inf
    out_tag_inf = f'/home/xiancai/DATA/FIRE_DATA/fire_{date}/temp_txt_tag_inf.txt' #标签分布信息存储地址
    if os.path.exists(out_tag_inf):
        os.remove(out_tag_inf)
    os.mknod(out_tag_inf)  #
    fp = open(out_tag_inf, mode="r+", encoding="utf-8")
    fp.write(str(tag_inf))
    fp.close()
    print(f'tag information: saved to {out_tag_inf}')
    print(tag_inf)

def remove_xxdate_from_classify_dataset():
    #
    input = '/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    ls=os.listdir(input)
    tag=0
    for i in ls:
        if i =='train.txt' or i =='test.txt':
            continue
        da=i.split('_d')[1][:5]
        if da==date:
            os.remove(input+i)
            tag+=1
            print(f'{tag}/{len(ls)}:{input+i} removed')


def make_fire_classify_dataset_metrix():
    '''
    制作度量识别训练验证集
    :return:
    '''
    input='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    output='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset_metrix/'
    output_tra=f'{output}train.txt'
    output_val = f'{output}test.txt'

    ls = os.listdir(input)
    res = []
    for i in ls:
        if i == 'train.txt' or i == 'test.txt':
            continue
        x = i  # 图片地址
        y = i.split('_')[0]  # 类别
        res.append(x + ' ' + y)

    # 划分 各类随机抽~%
    # flag=2
    ls_img_tra = []
    ls_img_val = []
    hm_img = dict()  # k:类别 v:文件名列表
    for i in res:
        ty = i.split(' ')[-1]  # 取前缀为类别
        if not hm_img.get(ty):
            hm_img[ty] = [i]
        else:
            hm_img[ty].append(i)
    for k in hm_img.keys():
        random.shuffle(hm_img[k])
        flag = max(int(0.1 * len(hm_img[k])), 1)  # 各类别~%
        ls_img_val += hm_img[k][:flag]
        ls_img_tra += hm_img[k][flag:]

    random.shuffle(ls_img_val)
    random.shuffle(ls_img_tra)  # 打乱顺序，一个batch内的数据类别更加丰富 重要！！！
    with open(output_tra, 'w') as f:
        for i in ls_img_tra:
            f.write(i + '\n')

    # metrix验证集 [im0,im1,是否同类]
    with open(output_val, 'w') as f:
        for i in ls_img_val:
            im0,la0=i.split(' ')
            for j in ls_img_val:
                im1,la1=j.split(' ')

                # 均衡
                if im0 == im1:
                    continue
                if la0!=la1 and random.random()>1/113:
                    continue
                f.write(im0+' '+ im1 + ' ' + str(int(la0==la1)) + '\n') #

    print(f'total:{len(ls) - 2}')
    print(f'saved to {output_tra} :{len(ls_img_tra)} and {output_val}:{len(ls_img_val)}')

def make_fire_classify_dataset_metrix_remove():
    '''
    制作度量识别训练验证集,不包括某个日期数据
    :return:
    '''
    remove_date='12_03'
    input='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset/'
    output='/home/xiancai/DATA/FIRE_DATA/fire_classify_dataset_metrix/'
    output_tra=f'{output}train_no_{remove_date}.txt'
    output_val = f'{output}test_no_{remove_date}.txt'

    ls = os.listdir(input)
    res = []
    for i in ls:
        if i == 'train.txt' or i == 'test.txt':
            continue

        da = i.split('_d')[1][:5]
        print(f'debug: date:{da}')
        if da ==remove_date:
            continue
        x = i  # 图片地址
        y = i.split('_')[0]  # 类别
        res.append(x + ' ' + y)

    # 划分 各类随机抽~%
    # flag=2
    ls_img_tra = []
    ls_img_val = []
    hm_img = dict()  # k:类别 v:文件名列表
    for i in res:
        ty = i.split(' ')[-1]  # 取前缀为类别
        if not hm_img.get(ty):
            hm_img[ty] = [i]
        else:
            hm_img[ty].append(i)
    for k in hm_img.keys():
        random.shuffle(hm_img[k])
        flag = max(int(0.1 * len(hm_img[k])), 1)  # 各类别~%
        ls_img_val += hm_img[k][:flag]
        ls_img_tra += hm_img[k][flag:]

    random.shuffle(ls_img_val)
    random.shuffle(ls_img_tra)  # 打乱顺序，一个batch内的数据类别更加丰富 重要！！！
    with open(output_tra, 'w') as f:
        for i in ls_img_tra:
            f.write(i + '\n')

    # metrix验证集 [im0,im1,是否同类]
    with open(output_val, 'w') as f:
        for i in ls_img_val:
            im0,la0=i.split(' ')
            for j in ls_img_val:
                im1,la1=j.split(' ')

                # 均衡
                if im0 == im1:
                    continue
                if la0!=la1 and random.random()>1/113:
                    continue
                f.write(im0+' '+ im1 + ' ' + str(int(la0==la1)) + '\n') #

    print(f'total:{len(ls) - 2}')
    print(f'saved to {output_tra} :{len(ls_img_tra)} and {output_val}:{len(ls_img_val)}')
if __name__ == '__main__':
    # check_fire_classify_dataset()
    # check_yolotojson()

    '''
    处理原始数据，得到检测和识别数据集
    '''
    # date='12_03'
    # process_fire_11_23() #
    # get_txt_from_xml()
    # # # remove_xxdate_from_fire_detect_dataset()
    # add_fire_11_23_to_firecontrol()
    #
    # #
    # changeImages()
    # # remove_xxdate_from_classify_dataset()
    # add_classify_dataset()
    # set_classify_tra_val()
    # # # print(date)



    make_fire_classify_dataset_metrix_remove()