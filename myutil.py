'''
数据处理工具
'''
import glob
import os
import random
import shutil
import cv2
import argparse

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

def changeImages():
    #抠图

    input_img='/home/xiancai/Ruler/Pytorch/firecontrol/images/train/'
    input_lab='/home/xiancai/Ruler/Pytorch/firecontrol/labels/train/'
    out_img='/home/xiancai/firecontrol_class_data_2/'
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

def setResnetLabel():
    #配置消防识别数据集标签 resnet
    input='/home/xiancai/firecontrol_class_data_2/*/*'
    output_tra='/home/xiancai/Ruler/pytorch-image-classfication/data/train.txt'
    output_val='/home/xiancai/Ruler/pytorch-image-classfication/data/test.txt'
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
        y=int(name[-1].split('_')[0])-1#类别
        res.append(x+' '+str(y))
    random.shuffle(res)
    size_tra=int(len(res)*0.9)

    with open(output_tra, 'w') as f:
        for i in range(0,size_tra):
            f.write(res[i] + '\n')
    with open(output_val, 'w') as f:
        for i in range(size_tra,len(res)):
            f.write(res[i] + '\n')









if __name__ == '__main__':
    changeImages()
    # read_anchors()
    # divide_ruler()
    # counts_files()


    # python3 /home/xiancai/Ruler/Pytorch/myutil.py --option moveAtoB --A '/home/xiancai/test/ima1/*' --B '/home/xiancai/test/ima2/' --buf 'bf3_'
    # python3 /home/xiancai/Ruler/Pytorch/myutil.py --option check_fal
    # input_add='/home/xiancai/no_ruler_0916/'
    # out_add='/home/xiancai/915_data_yolov5_s/images/train/'
    # addFalseSamples(input_add,out_add)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--A', type=str, default='/home/xiancai/test/ima2/*', help='目录+模式')
    # parser.add_argument('--B', type=str, default='/home/xiancai/test/ima1/', help='目录')
    # parser.add_argument('--buf', type=str, default='buf2_', help='前缀')
    # parser.add_argument('--option', type=str, default='moveAtoB', help='')
    # hm={'copyAtoB':copyAtoB,'moveAtoB':moveAtoB,'check_fal':check_fal,'divide_jsonimage':divide_jsonimage}
    # opt = parser.parse_args()
    # if opt.option=='check_fal':
    #     hm[opt.option]()
    # else:
    #     hm[opt.option](opt.A,opt.B,opt.buf)





# import torch
# import cv2
# from backbones.resnet import resnet18
# from data.dataset import Dataset, DataLoaderX
#
# #载入模型
# pytorch_model = 'checkpoints/resnet18/resnet18-13-best.pth'
# model=resnet18()#模型结构
# m_dir=torch.load(pytorch_model)
# model.load_state_dict(m_dir,False)
# model.eval()
#
# # 载入测试图片
# data='/home/xiancai/Ruler/pytorch-image-classfication/data/test.txt'
# root='/home/xiancai/Ruler/pytorch-image-classfication/data/'
# with open(data) as f:
# 	ls=f.readlines()
# 	im_labs=[]
# 	for i in ls:
# 		im_labs.append(i.strip().split(' '))
# # valset = Dataset(root_dir='data', data_list='data/test.txt', local_rank=0)
# # val_loader  = DataLoaderX(local_rank=0, dataset=valset, batch_size=1, num_workers=2, pin_memory=True, drop_last=True)
#
# # test
# for img_add,lab in im_labs:
# 	img=cv2.imread(root+img_add)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img=cv2.resize(img,(112,112))
# 	#
# 	pre=model(img)
# 	print(pre)
#
# # for images, labels in val_loader:
# # 	predicts = model(images)
# # 	print(predicts)



