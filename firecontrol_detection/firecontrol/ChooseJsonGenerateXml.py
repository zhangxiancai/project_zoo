#-*-coding:utf-8 -*-
import json
import glob
import os,shutil
import sys
from xml.dom import minidom

def judge_folder_exists(Folder,num=0):
    '''
    input  : folder dir,0 or 1 or 2
    output : No
    role   : judge folder exist or no exist,create folder/file
    '''
    if not os.path.exists(Folder):
        if num == 1:
            os.makedirs(Folder)
        elif num == 2:
            os.mknod(Folder)
        else:
            exit('no find %s'%Folder)

def ProgressBar(TotalNum,Num):
    '''
    input:total number,new finish number
    output:output now progress bar
    role:output progress bar 
    '''
    Progress = (Num+1) * 100.0 / TotalNum
    Output = '>'*int(Progress) + ' '*(100 - int(Progress))
    sys.stdout.write('\r' + Output + '%s%%'%Progress)
    sys.stdout.flush()

def GetFile(FileDir):
    '''
    input:file directory 
    output:file(have path)
    role:read file directory,get file(have path)
    '''
    FilePath=[]
    FilePath=glob.glob(FileDir+'/*')#
    return FilePath

def CopyFile(OldFilePath,NewFilePath):
    '''
    input:old file path,new file path
    ouput:No
    role:copy file into new path
    '''
    shutil.copy(OldFilePath,NewFilePath)

def SaveXml(filename,w,h,Value,SaveXmlDir):
    '''
    input:image name,image width,iamge height,x y w h value,save xml directory
    output:No
    role:generate xml file
    '''
    dom=minidom.Document()
    root=dom.createElement('annotation')
    dom.appendChild(root)
    Name_1=filename
    Filename=dom.createElement('filename')
    root.appendChild(Filename)
    FilenameText=dom.createTextNode(str(Name_1))
    Filename.appendChild(FilenameText)

    SizeWidth=w
    SizeHeight=h
    Size=dom.createElement('size')
    root.appendChild(Size)
    Width=dom.createElement('width')
    Size.appendChild(Width)
    WidthText=dom.createTextNode(str(SizeWidth))
    Width.appendChild(WidthText)
    Height=dom.createElement('height')
    Size.appendChild(Height)
    HeightText=dom.createTextNode(str(SizeHeight))
    Height.appendChild(HeightText)
    Depth=dom.createElement('depth')
    Size.appendChild(Depth)
    DepthText=dom.createTextNode('3')
    Depth.appendChild(DepthText)

    Judge = 0
    Total = 0
    for Num in range(len(Value) // 5):
        Total = Total + 1
        Type  = Value[Num*5 + 0]
        W=float(Value[Num*5 + 3])
        H=float(Value[Num*5 + 4])
        Xmin1=float(Value[Num*5 + 1])
        if Xmin1 < 0:
            Xmin1 = 0
        elif Xmin1 >= int(SizeWidth):#
            continue

        Xmax1=W+Xmin1
        if Xmax1 > int(SizeWidth):
            Xmax1=SizeWidth
        elif Xmax1 < 0:
            continue

        Ymin1=float(Value[Num*5 + 2])
        if Ymin1 < 0:
            Ymin1 = 0
        elif Ymin1 >= int(SizeHeight):
            continue

        Ymax1=H+Ymin1
        if Ymax1 >int(SizeHeight):
            Ymax1 = SizeHeight
        elif Ymax1 < 0:
            continue

        if Xmin1 == Xmax1 or Ymin1 == Ymax1:
            print('boxs error',Name_1)
            Judge = Judge + 1
            continue

        Object=dom.createElement('object')
        root.appendChild(Object)

        Name=dom.createElement('name')
        Object.appendChild(Name)
        NameText=dom.createTextNode(str(Type))
        Name.appendChild(NameText)

        Bndbox=dom.createElement('bndbox')
        Object.appendChild(Bndbox)

        Xmin=dom.createElement('xmin')
        Bndbox.appendChild(Xmin)
        XminText=dom.createTextNode(str(int(Xmin1)))
        Xmin.appendChild(XminText)

        Ymin=dom.createElement('ymin')
        Bndbox.appendChild(Ymin)
        YminText=dom.createTextNode(str(int(Ymin1)))
        Ymin.appendChild(YminText)

        Xmax=dom.createElement('xmax')
        Bndbox.appendChild(Xmax)
        XmaxText=dom.createTextNode(str(int(Xmax1)))
        Xmax.appendChild(XmaxText)

        Ymax=dom.createElement('ymax')
        Bndbox.appendChild(Ymax)
        YmaxText=dom.createTextNode(str(int(Ymax1)))
        Ymax.appendChild(YmaxText)

    if Judge != Total :
        image_suffix = os.path.splitext(filename)[-1]
        xml_file=SaveXmlDir+'/'+Name_1.replace(image_suffix,'.xml')
        with open(xml_file,'w') as file:
            dom.writexml(file,indent='',addindent='\t',newl='\n',encoding='UTF-8')

def ReadJsonFile(JsonFilepath,ImageDir,SaveXmlDir,SaveImageDir):
    '''
    input:single json file path,image directory,save xml directory,save new image directory
    output:No
    role:read json file,move error file 
    '''
    with open(JsonFilepath,'r',encoding='UTF-8') as JsonFile:
        Json=json.load(JsonFile)
    Faces        = Json['faces']
    Filename     = os.path.basename(JsonFilepath).replace('.xml','')

    W = Json['image_width']
    H = Json['image_height']
    Num=len(Faces)
    Value=[]
    for face in range(Num):
        Type1  = Faces[face]['type']
        print('label is:{}'.format(Type1))
        if Type1   == '1表面脏污':
            Type = 'printing_dirt'
        elif Type1   == '2波浪':
            Type = 'wave'
        elif Type1 == '3残缺':
            Type = 'incomplete_printing'
        elif Type1 == '4表面损伤' :
            Type = 'scratch'
        elif Type1 == '5烤焦起泡' :
            Type = 'paint_burnedbubble'
        elif Type1 == '6烤漆色差_青线割线' :
            Type = 'painting_discolor__green_line'
        elif Type1 == '7空白接头' :
            Type = 'blank_tape'
        elif Type1 == '8偏位' :
            Type = 'offset'
        elif Type1 == '9前工序接头' :
            Type = 'previous_connector'
        elif Type1 == '11印刷重' or Type1 == '11印刷重z' or Type1=='11轻重':
            Type = 'printing_heavy'
        elif Type1 == '12印刷轻' :
            Type = 'printing_light'
        elif Type1 == '13上光起泡':
            Type = 'lacquering_bubble'
        elif Type1 == '14黑块':
            Type = 'black_piece'
        else:
            print('label is error :{}'.format(Type1))
            print('filename:{}'.format(Filename))
            exit()
        Value.append(Type)
        x = Faces[face]['x']
        Value.append(x)
        y = Faces[face]['y']
        Value.append(y)
        w = Faces[face]['w']
        Value.append(w)
        h = Faces[face]['h']
        Value.append(h)
    if len(Value):    
        ImageFilePath = os.path.join(ImageDir,Filename)
        if os.path.exists(ImageFilePath):
            CopyFile(ImageFilePath,SaveImageDir)
            SaveXml(Filename,W,H,Value,SaveXmlDir)
        else:
            print('no such {} image file'.format(ImageFilePath))

if __name__ == '__main__':
    # JsonDir      = '../Data/Train/printing_dirt/new/Json'
    # ImageDir     = '../Data/Train/printing_dirt/new/Image_json'
    # SaveXmlDir   = '../Data/Train/printing_dirt/new/Annotations'
    # SaveImageDir = '../Data/Train/printing_dirt/new/JPEGImages'

    # JsonDir      = '/home/xiancai/test/lab'
    # ImageDir     = '/home/xiancai/test/ima'
    # SaveXmlDir   = '/home/xiancai/test/voc/Annotations'
    # SaveImageDir = '/home/xiancai/test/voc/JPEGImages'

    JsonDir      = '/home/xiancai/firecontrol_data_1/1/1/json'
    ImageDir     = '/home/xiancai/firecontrol_data_1/1/1/image'
    SaveXmlDir   = '/home/xiancai/firecontrol_data_1_voc/1/Annotations'
    SaveImageDir = '/home/xiancai/firecontrol_data_1_voc/JPEGImages'
    judge_folder_exists(JsonDir)
    judge_folder_exists(ImageDir)
    judge_folder_exists(SaveXmlDir,1)
    judge_folder_exists(SaveImageDir,1)

    JsonFilePath = GetFile(JsonDir)
    TotalNum     = len(JsonFilePath)
    for Num , JsonFilepath in enumerate(JsonFilePath):
        ProgressBar(TotalNum,Num)
        ReadJsonFile(JsonFilepath,ImageDir,SaveXmlDir,SaveImageDir)

    print('xml个数：', len(os.listdir(SaveXmlDir)))
    print('image个数：', len(os.listdir(SaveImageDir)))