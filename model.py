import torch
from torch import nn
import torch.nn.functional as F
import math
from backbone.repvgg import get_RepVGG_func_by_name
from backbone.se_block import SEBlock_v2
import rep_utils


class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True,export_onnx=False,teacher=False,student=False,use_gpu=True):
        super(SixDRepNet, self).__init__()

        self.teacher = teacher # 是否为teacher网络
        self.student = student # 是否为student网络
        # BACKBONE
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt,strict=False)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4

        # POOL
        self.export_onnx = export_onnx
        if not self.export_onnx:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            self.gap = nn.AvgPool2d(kernel_size=2) # 244:7  112:4

        # SE
        if backbone_name=='RepVGG-A0s-se':
            self.se = SEBlock_v2(1280,1280//16)
        else:
            self.se= nn.Identity()

        # LINEAR
        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels
        fea_dim = last_channel
        self.linear_reg = nn.Linear(fea_dim, 6)

        self.use_gpu = use_gpu

    def forward(self, x):
        if self.teacher: # 如果为teacher网络
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        if self.student:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            xf = self.layer4(x)
            x = self.gap(xf)
            x = self.se(x)
            x = torch.flatten(x, 1)

            # x = self.drop(x)
            x = self.linear_reg(x)
            return rep_utils.compute_rotation_matrix_from_ortho6d(x,self.use_gpu), xf
        if self.export_onnx: # 如果导出onnx模型
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.gap(self.gap(x))
            x = self.se(x)
            x = torch.flatten(x, 1)

            x = self.linear_reg(x)
            return x

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.se(x)
        x = torch.flatten(x, 1)

        # x = self.drop(x)
        x = self.linear_reg(x)
        return rep_utils.compute_rotation_matrix_from_ortho6d(x,self.use_gpu)

# class A0_se(SixDRepNet):



class SixDRepNet2(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      


        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = rep_utils.compute_rotation_matrix_from_ortho6d(x)

        return out


from backbone.resnet import resnet18
class Resnet18_face_angle(nn.Module):
    def __init__(self,export_onnx=False):
        super(Resnet18_face_angle, self).__init__()
        self.res18 = resnet18(num_classes=6) # input_size 112*112
        self.export_onnx = export_onnx

    def forward(self, x):

        x = self.res18(x)
        if self.export_onnx: # 如果导出onnx模型
            return x
        else:
            return rep_utils.compute_rotation_matrix_from_ortho6d(x)

# from torchvision import models
# models.resnet18(pretrained=True)