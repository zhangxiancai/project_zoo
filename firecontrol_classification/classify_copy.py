import torch
import cv2
from backbones.resnet import resnet18
from utils.net_utils import load_model


# 载入模型
# pytorch_model = 'checkpoints/resnet18/resnet18-68-best.pth'
# model = resnet18()  # 模型结构
# m_dir = torch.load(pytorch_model)
# model.load_state_dict(m_dir,False)
# model.eval()
model=resnet18()
load_model(model,'checkpoints/resnet18/resnet18-68-best.pth')
model.eval()


def classify(img_address):
    #预测
    img = cv2.imread(img_address)
    if img is None:
        return -1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = torch.from_numpy(img).float()
    img /= 255.0
    img=(img-0.5)/0.5
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(img.shape)
    print(img.permute(0,3,1,2).shape)#
    img=img.permute(0, 3, 1, 2)
    # img=torch.ones(1,3,112,112)
    pre = model(img)
    print(pre)
    cls=torch.argmax(pre)
    return cls


if __name__ == '__main__':
    cls = classify('/home/xiancai/Ruler/pytorch-image-classfication/data/11/11_WIN_20201229_13_27_29_Pro.jpg')
    print(cls)

