
#json数据集转yolov5数据集
import os
import json
import re
import shutil


def single_jsontotxt(json_dir,out_dir):
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
    file_str=''
    for t in content['faces']:
        # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
        if t['x']<0:
            t['x']=0
        if t['x']>content['image_width']:
            t['x']=content['image_width']

        if t['x'] +t['w']>content['image_width']:
            t['w']=content['image_width']-t['x']

        if t['y'] < 0:
            t['y'] = 0
        if t['y'] > content['image_height']:
            t['y'] = content['image_height']
        if t['y'] + t['h'] > content['image_height']:
            t['h'] = content['image_height'] -t['x']


        x = (t['x']+t['w']/2) /content['image_width']#
        y = (t['y']+t['h']/2)/ content['image_height']
        w = t['w'] /content['image_width']
        h = t['h']/ content['image_height']
        # type=re.sub('\D','',str(t['type']))#截取typeid
        type = int(str(t['type']).split('_')[0])-1  # 截取typeid
        type=0#只检测
        # file_str +=str(typeids_jsontotxt[int(type)]) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + ' ' + str(round(h, 6))+'\n'
        file_str +=str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + ' ' + str(round(h, 6)) + '\n'

        # 记录数量
        if not hm.get(type):
            hm[type]=0
        hm[type]+=1

    if  os.path.exists(filename):
        os.remove(filename)
    os.mknod(filename)#
    fp = open(filename, mode="r+", encoding="utf-8")
    fp.write(file_str[:-1])
    fp.close()


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

    print(json_data_dir,t1,t2)#t1xml, t2jpg
    imas=os.listdir(out_dir_image)
    labs=os.listdir(out_dir_label)
    for i in imas:
        if i+'.xml' not in labs:
            print('image:'+i+' 无标签')

if __name__ == '__main__':
    # input_dir="/home/xiancai/firecontrol_data_1/1/"
    # ls=os.listdir(input_dir)
    # for i in ls:
    #     divide_jsonimage(input_dir+i+'/')

    ls=os.listdir('/home/xiancai/firecontrol_data_1/1/')
    ls.sort()
    for i in ls:
        json_data_dir = '/home/xiancai/firecontrol_data_1/1/'+i+'/'  # json格式数据集路径
        out_dir_label = '/home/xiancai/Ruler/Pytorch/firecontrol_com/labels/train/'  # 输出的 txt（label）文件路径
        out_dir_image='/home/xiancai/Ruler/Pytorch/firecontrol_com/images/train/'# 输出的图片路径

        typeids_jsontotxt= {}
        # typeids_jsontotxt = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 11: 8, 14: 9}
        hm=dict()#各个类别的样本数量
        ls=os.listdir(json_data_dir)
        ls.sort()
        # ls=ls[:100]
        if not os.path.exists(out_dir_label):
            os.makedirs(out_dir_label)
        if not os.path.exists(out_dir_image):
            os.makedirs(out_dir_image)
        t1,t2=0,0
        for it in ls:
            if it[-3:]=='xml':
                input_add=json_data_dir+it
                type_buf=json_data_dir.split(('/'))[-2]
                output_add=out_dir_label+type_buf+'_'+it[:-7]+'txt'
                single_jsontotxt(input_add,output_add)#
                t1+=1
            if it[-3:]=='jpg':
                type_buf=json_data_dir.split(('/'))[-2]
                output_add=out_dir_image+type_buf+'_'+it
                shutil.copy(json_data_dir+it,output_add)
                t2+=1
        print('type'+type_buf+':',len(ls))

        print('json_labels',t1)
        print('json_images:', t2)
        print('yolov5_labels:out_dir_label', len(os.listdir(out_dir_label)))
        print('yolov5_images:out_dir_image',len(os.listdir(out_dir_image)))

        print('类别数：',hm)
        # print(sum(hm.values()))
        # print('js_id:yo_id',typeids_jsontotxt)





# // {
# //     "list":["1_65水带","2_80水带","3_多功能消防水枪","4_黄色消防头盔",
# //         "5_空呼气瓶9L","6_灭火防护服","7_灭火防护手套","8_灭火防护靴",
# //         "9_灭火防护腰带","10_泡沫枪PQ6","11_抢险救援服","12_抢险救援头盔",
# //         "13_抢险救援靴","14_三分水器","15_水域救援头盔","16_水域救援靴",
# //         "17_水域漂浮背心","18_消防栓扳手","19_新款黄色消防头盔","20_液压剪",
# //         "21_液压救援顶杆","22_正压式呼吸面罩","23_直流水枪","24_止水器"
# // ],
# //     "default":"1_65水带"
# // }