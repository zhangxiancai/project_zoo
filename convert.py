import argparse
import os
from model import SixDRepNet
import torch
from backbone.repvgg import repvgg_model_convert

# parser = argparse.ArgumentParser(description='SixDRepNet Conversion')
# parser.add_argument('load',  help='path to the weights file',default='/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1648019395_bs64/RepVGG-A0s_epoch_25_mae8.4703.tar',type=str)
# parser.add_argument('save',  help='path to the weights file',default='/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1648019395_bs64/RepVGG-A0s_epoch_25_mae8.4703.pth')
# parser.add_argument('-a', '--arch',  default='RepVGG-A0s')


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def convert():
    # args = parser.parse_args()

    print('Loading model.')
    model = SixDRepNet(backbone_name=model_type,
                            backbone_file='',
                            deploy=False,
                            pretrained=False)

    # Load snapshot
    saved_state_dict = torch.load(tar)

    load_filtered_state_dict(model, saved_state_dict['model_state_dict'])
    print('Converting model.')
    repvgg_model_convert(model, save_path=pth)
    print('Done.')


tar='/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649297858_bs64/RepVGG-A0_epoch_177_mae13.5007.tar'
pth='/home/xiancai/face_angle/6DRepNet/output/snapshots/SixDRepNet_1649297858_bs64/RepVGG-A0_epoch_177_mae13.5007.pth'
model_type='RepVGG-A0'

if __name__ == '__main__':
    convert()