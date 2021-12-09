'''
消防物品分类，根据检测框
'''
import numpy as np
import torch
import cv2
import torch.nn as nn
import os


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes, width_mul=0.5 ):
        super().__init__()
        self.in_channels = int(64 * width_mul)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * width_mul), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * width_mul)),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, int(64 * width_mul), num_block[0], 2)
        self.conv3_x = self._make_layer(block, int(128 * width_mul), num_block[1], 2)
        self.conv4_x = self._make_layer(block, int(256 * width_mul), num_block[2], 2)
        self.conv5_x = self._make_layer(block, int(512 * width_mul), num_block[3], 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.dropout = nn.Dropout(p=0.4, inplace=True)  # 0.4
        self.fc = nn.Linear(int(512 * width_mul) * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc(output)

        return output


def resnet18(cls_number):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=cls_number)


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=False):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    # check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# 载入模型
model_address='/home/xiancai/fire-equipment-demo/firecontrol_classification/Result/2021_12_06/fire_classify_resnet18_cls114_12_06.pth'
cls_number=114

cls_names = {0: '65水带', 1: '80水带', 2: '多功能消防水枪', 3: '黄色消防头盔', 4: '空呼气瓶9L', 5: '灭火防护服',
             6: '灭火防护手套', 7: '灭火防护靴',8: '消防安全腰带', 9: '泡沫枪PQ6',10: '抢险救援服',
             11: '抢险救援头盔', 12: '抢险救援靴', 13: '三分水器', 14: '水域救援头盔', 15: '水域救援靴',
             16: '水域漂浮背心',17: '消防栓扳手', 18: '新款黄色消防头盔', 19: '液压剪',20: '液压救援顶杆',
             21: '正压式呼吸面罩', 22: '直流水枪', 23: '止水器', 24: '80公转80公转接头', 25: 'PQ8泡沫枪',
             26: 'PQ16泡沫枪',27: '防高温手套', 28: '隔热服',29: '灭火防护头套', 30: '轻型防化服',
             31: '消防员二类吊带', 32: '消防员接触式送受话器', 33: '消防员三类吊带', 34: '便携式移动消防灯组',35: '阀门堵漏套具',
             36: '非接触式红外测温仪', 37: '隔离警示带',38: '呼吸器背架', 39: '混凝土切割锯', 40: '捆绑式堵漏气垫',
             41: '炮口', 42: '无齿锯', 43: '消防专用救生衣', 44: '移动消防水炮',45: '6.8L正压式呼吸器',
             46: '大力剪', 47: '地下消火栓扳手', 48: '二分水器',49: '红色消防头盔(全盔)', 50: '黄色消防头盔(全盔)',
             51: '撬棍', 52: '铁锹', 53: '消防锤', 54: '消防尖斧', 55: '消防平斧',
             56: '滤水器',57:'80母转80公转接头',58:'冲击钻',59:'机动链锯',60:'闪光警示灯',
             61: '水质分析仪',62:'电动顶杆',63:'多功能担架',64:'钢丝绳速差器',65:'雷达生命探测仪',
             66: '外封式堵漏袋',67:'消防救生照明线',68:'小孔堵漏枪',69:'折叠式担架',70:'注入式堵漏工具',
             71: '背负式吸水雾', 72: '防静电内衣', 73: '呼救器后场接收装置', 74: '激光测距仪', 75: '木制堵漏楔',
             76: '热成像仪', 77: '手持扩音器', 78: '无火花工具', 79: '无线智能呼救器', 80: '消防阻燃毛衣',
             81: '防爆手持电台', 82: '过滤式消防自救呼吸器', 83: '混凝土切割锯片', 84: '救生抛投器', 85: '军事毒剂侦检仪',
             86: '灭火毯', 87: '手动破拆工具组', 88: '头骨通讯装置', 89: '无齿锯片', 90: '无线复合气体探测仪',
             91: '消防腰斧', 92: '消防员呼救器', 93: '医疗箱', 94: '移车器', 95: '移动照明灯',
             96: '有毒有害物质识别仪', 97: '自救安全绳(8毫米)', 98: '自喷荧光漆', 99: '避火服', 100: '便携式毁锁器',
             101: '单人洗消帐篷', 102: '手动液压千斤顶', 103: '双轮异向切割锯', 104: '四合一气体检测仪', 105: '液压开门器',
             106: '防爆照明灯', 107: '护目镜', 108: '警戒标志杆', 109: '救生缓降器', 110: '绝缘剪断钳',
             111: '抢险救援手套', 112: '危险警示牌', 113: '无人机'}

model = resnet18(cls_number)
load_model(model, model_address)  # accuary=0.9877
model.eval()

def changeImages(img_address, box):
    '''
    裁剪图片,调整为正方形
    :param img_address:
    :param box:xywh
    :return:
    '''

    img = cv2.imread(img_address)
    H, W = img.shape[0], img.shape[1]

    # ts=lab
    # check yolo标签
    x, y, w, h = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
    c1, c2 = max(x - int(w / 2), 0), min(x + int(w / 2), W)
    r1, r2 = max(y - int(h / 2), 0), min(y + int(h / 2), H)  # yolo标签越界修正（11/2425）
    x, y, w, h = (c1 + c2) // 2, (r1 + r2) // 2, c2 - c1, r2 - r1

    # 转为正方形
    s = int(max(w, h) / 2)
    c1, c2 = x - s, x + s
    r1, r2 = y - s, y + s

    # 越界填充
    if c1 < 0:
        pad = -c1
        img = cv2.copyMakeBorder(img, 0, 0, pad, 0, cv2.BORDER_CONSTANT, 0)
        c1, c2 = c1 + pad, c2 + pad
    if r1 < 0:
        pad = -r1
        img = cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
        r1, r2 = r1 + pad, r2 + pad
    if c2 > img.shape[1]:
        pad = c2 - img.shape[1]
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, 0)
    if r2 > img.shape[0]:
        pad = r2 - img.shape[0]
        img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0)

    img = img[r1:r2, c1:c2, ...]  # 截取
    return img


def classify(img_address, box):
    '''
    分类
    :param img_address: 图片地址(str)
    :param box: xywh
    :return: 类别（str）
    '''

    # 预处理
    image = changeImages(img_address, box)  # 裁剪图片,调整为正方形
    # cv2.imwrite('扣图.jpg',image)# debug
    resized = cv2.resize(image, (112, 112))
    resized = resized[..., ::-1]  # BGR to RGB
    resized = resized.swapaxes(1, 2).swapaxes(0, 1)
    resized = np.reshape(resized, [1, 3, 112, 112])
    resized = np.array(resized, dtype=np.float32)
    resized = (resized - 127.5) / 128.0
    # resized = (resized - 127.5) / 127.5
    img = torch.from_numpy(resized)

    # inference
    pre = model(img)
    cls = torch.argmax(pre)
    return cls_names[int(cls)]


if __name__ == '__main__':
    # lab={'x': 316.30667, 'y': 468.55704, 'w': 793.00353, 'h': 283.41781}
    # x,y,w,h=(lab['x'] + lab['w'])/2/1280,(lab['y'] + lab['h'])/2/960,lab['w']/1280,lab['h']/960

    cls = classify('test1.jpg', [0.554296875, 0.6354166666666666, 0.60390625, 0.3020833333333333])  # 分类
    print(cls)

    # # 载入测试图片集
    # data = '/home/xiancai/Ruler/pytorch-image-classfication/data/test.txt'
    # root = '/home/xiancai/Ruler/pytorch-image-classfication/data/'
    # with open(data) as f:
    #     ls = f.readlines()
    #     im_labs = []  # 图片+标签
    #     for i in ls:
    #         im_labs.append(i.strip().split(' '))
    #
    # res = np.zeros((24,24),int) #分类结果
    # errs=[]
    # for img_addr, lab in im_labs:
    #     lab=int(lab)
    #     cls = classify(root + img_addr)
    #     cls=int(cls)
    #     res[lab][cls]+=1
    #     if lab!=cls:
    #         errs.append([lab,cls])
    # print(res)
    # print('errors:',errs)
