import glob
import xml.etree.ElementTree as ET
import os
import shutil

def JudgeFolderExists(Folder,num=0):
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

def CopyFile(OldFilePath,NewFilePath):
    '''
    input  : old file path,new file path
    output : No
    role   : copy file into new path
    '''
    shutil.copy(OldFilePath,NewFilePath)   

def GetFile(FileDir):
    '''
    input:file directory 
    output:file file (have path)
    role:read file directory,get file 
    '''
    FilePath=glob.glob(FileDir+'/*')
    return FilePath

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def getXml(Annotations,Classes,SaveTxtFile,Dic):
    Judge = False
    with open(SaveTxtFile,'w') as F:
        root    = ET.parse(Annotations).getroot()
        objects = root.findall('object')
        Size    = root.find('size')
        w       = int(Size.find('width').text.strip())
        h       = int(Size.find('height').text.strip())
        bboxs   = []
        for obj in objects:
            bbox = obj.find('bndbox')
            ClassName = obj.find("name").text.lower().strip()
            if ClassName not in Dic:
                Dic[ClassName] = 1
            else:
                Dic[ClassName] += 1

            if ClassName == 'printing_dirt':
                class_id = 0
            elif ClassName == 'wave':
                class_id = 1
            elif ClassName == 'incomplete_printing':
                class_id = 2
            elif ClassName == 'scratch':
                class_id = 3
            elif ClassName == 'paint_burnedbubble':
                class_id = 4
            elif ClassName == 'painting_discolor__green_line':
                class_id = 5
            elif ClassName == 'blank_tape':
                class_id = 6
            elif ClassName == 'offset':
                class_id = 7
            elif ClassName == 'previous_connector':
                class_id = 8
            elif ClassName == 'printing_heavy':
                class_id = 9
            elif ClassName == 'printing_light':
                continue
            elif ClassName == 'lacquering_bubble':
                class_id = 10
            elif ClassName == 'black_piece':
                class_id = 11
            else:
                print('error label:{}'.format(Annotations))
                exit()
            xmin = float(bbox.find('xmin').text.strip())
            ymin = float(bbox.find('ymin').text.strip())
            xmax = float(bbox.find('xmax').text.strip())
            ymax = float(bbox.find('ymax').text.strip())
            b = (xmin, xmax, ymin, ymax)
            bb = convert((w,h), b)
            F.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            Judge = True
    if os.path.getsize(SaveTxtFile) == 0 :
        os.system('rm '+SaveTxtFile)
    return Judge

if __name__ == '__main__':
    # Root         = '../Data/Test'
    # folder       = ['4_16_collect/old',
    #                 'blank_tape/new','blank_tape/old',
    #                 'incomplete_printing/old','incomplete_printing/new',
    #                 'lacquering_bubble/old',
    #                 'offset/old',
    #                 'paint_burnedbubble/old','paint_burnedbubble/new',
    #                 'painting_discolor__green_line/old','painting_discolor__green_line/new',
    #                 'previous_connector/old','previous_connector/new',
    #                 'printing_dirt/old','printing_dirt/new',
    #                 'printing_heavy/old','printing_heavy/new',
    #                 'scratch/old',
    #                 'wave/old','wave/new']
    # SaveImageDir = 'images/val'
    # SaveTxtDir   = 'labels/val'

    Root         = '/home/xiancai'
    folder       = ['915_data_voc']
    SaveImageDir = '/home/xiancai/Ruler/ruler/images/train'
    SaveTxtDir   = '/home/xiancai/Ruler/ruler/labels/train'
    JudgeFolderExists(SaveImageDir,1)
    JudgeFolderExists(SaveTxtDir,1)

    Classes = ['printing_dirt'     ,'wave'                         ,'incomplete_printing','scratch',
               'paint_burnedbubble','painting_discolor__green_line','blank_tape'         ,'offset',
               'previous_connector','printing_heavy'               ,'printing_light'     ,'lacquering_bubble',
               'black_piece']
    Dic = dict()

    for j in folder:
        ImgFiles = GetFile(os.path.join(Root,j,'JPEGImages'))
        for JPEGImages in ImgFiles:
            Name         = os.path.splitext(os.path.basename(JPEGImages))[0]
            # image_suffix = os.path.splitext(os.path.basename(JPEGImages))[-1]
            Annotations  = os.path.join(Root,j,'Annotations/{}.xml'.format(Name))
            ImageName    = '{}_{}'.format(j.replace('/','_'),os.path.basename(JPEGImages))

            SaveTxtFile = os.path.join(SaveTxtDir,'{}.txt'.format(os.path.splitext(ImageName)[0]))
            Judge = getXml(Annotations,Classes,SaveTxtFile,Dic)
            if Judge:
                CopyFile(JPEGImages,os.path.join(SaveImageDir,ImageName))
    print("Dic:",Dic)
