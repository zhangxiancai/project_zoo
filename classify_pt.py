import cv2
import numpy as np

from model import SixDRepNet,Resnet18_face_angle
import rep_utils
import torch
#
# 测试脚本：/home/xiancai/face_angle/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/myutil_v3.py


class inference:

    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_05/RepVGG-A0s_epoch_180_mae8.4394.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_03_16/_epoch_30.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_03_23/_epoch_173_mae7.536.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/b/_epoch_30_mae7.473.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_06_transfer/RepVGG-A0s_epoch_180_mae8.5346.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_07/RepVGG-A0s_epoch_180_mae8.2222.pth'
    pt_path = '/home/xiancai/face_angle/6DRepNet/results/2022_04_07/RepVGG-A0s_epoch_180_mae8.0871_transfer.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649256873_bs64/RepVGG-A0s_epoch_1_mae7.1979.pth'
    # pt_path ='/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649256006_bs64/RepVGG-A0s_epoch_180_mae8.3902.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649298134_bs64/RepVGG-A0_epoch_165_mae7.7404.pth'
    # pt_path = '/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649297858_bs64/RepVGG-A0_epoch_177_mae13.5007.pth'

    img_size=(112, 112) # h w
    model = SixDRepNet(backbone_name='RepVGG-A0s',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)


    # pt_path = '/home/xiancai/face_angle/6DRepNet/results/other/6DRepNet_300W_LP_AFLW2000.pth'
    # img_size=(224, 224) # h w
    # model = SixDRepNet(backbone_name='RepVGG-B1g2',
    #                    backbone_file='',
    #                    deploy=True,
    #                    pretrained=False)

    # Load snapshot
    saved_state_dict = torch.load(pt_path, map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    gpu=0
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    print(f'face angle model:{pt_path}')

    def classify(self, img):

        img= cv2.imread(img) if isinstance(img,str) else img
        img = img[...,::-1] # to rgb
        img = cv2.resize(img, self.img_size[::-1])
        img = img /255.0
        img = img.transpose(2, 0, 1) # to c*h*w
        img = torch.from_numpy(img).type(torch.FloatTensor)

        mea=torch.tensor((0.485, 0.456, 0.406)).reshape(3, 1, 1).expand((3, self.img_size[0], self.img_size[1]))
        std=torch.tensor((0.229, 0.224, 0.225)).reshape(3, 1, 1).expand((3, self.img_size[0], self.img_size[1]))
        # std = torch.tensor((0.225, 0.225, 0.225)).reshape(3, 1, 1).expand((3, self.img_size[0], self.img_size[1]))
        img = (img - mea) / std
        img = img[None,...]  # to 1*c*h*w
        img = torch.Tensor(img).cuda(self.gpu)


        R_pred = self.model(img)

        euler = rep_utils.compute_euler_angles_from_rotation_matrices(
            R_pred) * 180 / np.pi
        p_pred_deg = euler[:, 0].cpu().item()
        y_pred_deg = euler[:, 1].cpu().item()
        r_pred_deg = euler[:, 2].cpu().item()

        return p_pred_deg,y_pred_deg,r_pred_deg



if __name__=='__main__':
    img='/data1/xiancai/FACE_ANGLE_DATA/other/test/image5.jpg'

    res=inference().classify(img)
    print(res)

