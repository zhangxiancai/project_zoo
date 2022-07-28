# 单层车牌识别

- 参考仓库：https://github.com/sirius-ai/LPRNet_Pytorch 

- 飞书文档： https://kr2ubu2yby.feishu.cn/docs/doccngyLFhZHYSpeZ7ExLB9uZeh

## Features
- 训练数据40w+
- 自定义数据增强：更改车牌背景颜色,车牌生成,模拟光斑
- loss：focal+ctc
- 支持ddp训练
- 丰富的测试脚本


# 仓库代码说明
- 仓库： https://gitlab.deepcam.cn/zhang/LPRNet_Pytorch.git 
- 本地路径：172.20.2.12:/home/xiancai/plate/LPRNet_Pytorch/
- 数据地址：/data1/xiancai/PLATE_DATA/
## 代码目录
```
.
├── chinese_license_plate_generator # 车牌生成库，
│   ├── augment.py # 数据增强脚本，生成黄牌，蓝绿牌，白牌
├── classify_plate_pt.py # pt模型推理和测试脚本
├── classify_plate.py # onnx模型推理和测试脚本
├── data
│   ├── load_data.py # 
├── log # 日志目录
├── loss 
│   ├── focal_ctc.py
├── model
│   ├── __init__.py
│   ├── LPRNet0.py
│   ├── LPRNet.py # 单层车牌识别模型结构文件
│   └── __pycache__
├── myutil.py # 车牌识别数据处理，
├── optim
├── pytorch2onnx.py # 转模型脚本
├── README.md
├── Result # pt和onnx模型保存目录
├── train_ddp.sh
├── train_dp.sh
├── train_eval.py
├── train_LPRNet_ddp.py # 训练脚本（ddp方式训练）
├── train_LPRNet_dp.py # 训练脚本（dp方式训练）
└── weights # checkpoint
```



## 处理数据
- 处理神目和ccpd等格式数据，扣取车牌，制作为车牌识别数据集
```
cd /home/xiancai/plate/LPRNet_Pytorch/
python3 myutil.py 
```


## 训练
- 多gpu， 使用DDP方式训练
```
cd /home/xiancai/plate/LPRNet_Pytorch/
sh train_ddp.sh
# nohup python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 train_LPRNet_ddp.py --sync-bn >log/train_d05_21.log 2>&1 &

```


## 转模型
- pt模型转为onnx模型
```
cd /home/xiancai/plate/LPRNet_Pytorch/
python3 pytorch2onnx.py
```



## 测试
- 测试pt模型，统计精度
```
cd /home/xiancai/plate/LPRNet_Pytorch/
python3 classify_plate_pt.py
```


- 测试onnx模型，统计精度
```
cd /home/xiancai/plate/LPRNet_Pytorch/
python3 classify_plate.py
```


