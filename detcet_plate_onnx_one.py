from detect_plate_onnx import detect_one_onnx,xyxy2xywh,show_results
import cv2
import torch

img_path='/data1/xiancai/PLATE_DATA/other/微信图片_20220123175140.jpg'
save_path='/data1/xiancai/PLATE_DATA/other/detect.jpg'
# detect
orgimg=cv2.imread(img_path)
det=detect_one_onnx(orgimg)
# 画图
gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
for j in range(det.shape[0]):
    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()  # to xywh 原图ratio
    conf = det[j, 4].cpu().numpy()
    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()  # to 原图ratio
    class_num = det[j, 15].cpu().numpy()
    orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
# save
cv2.imwrite(save_path, orgimg)
