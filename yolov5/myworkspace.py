import glob

import cv2

def ttest_mult():
    img_glob='/data1/xiancai/BABY_DATA/baby_detect_yolov5_05_17/images/val/*'
    ls=glob.glob(img_glob)
    for ind, i in enumerate(ls):
        img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        cv2.imwrite(f'/data1/xiancai/BABY_DATA/other/test_05_26/clahe{ind}.jpg', img_clahe)

        img_he = cv2.equalizeHist(img) # 直方图均衡化
        cv2.imwrite(f'/data1/xiancai/BABY_DATA/other/test_05_26/he{ind}.jpg', img_he)


if __name__=='__main__':

    ttest_mult()

    # img=cv2.imread('/data1/xiancai/BABY_DATA/baby_detect_yolov5_05_17/images/val/d05_17_1650889473373283_5.jpg',cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite('/data1/xiancai/BABY_DATA/other/test_05_26/origin.jpg', img)
    # clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # img_clahe=clahe.apply(img)
    # cv2.imwrite('/data1/xiancai/BABY_DATA/other/test_05_26/clahe.jpg',img_clahe)
    #
    # img_he = cv2.equalizeHist(img)
    # cv2.imwrite('/data1/xiancai/BABY_DATA/other/test_05_26/he.jpg', img_he)
