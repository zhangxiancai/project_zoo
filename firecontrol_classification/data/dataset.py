import numbers
import os
import queue as Queue
import threading

import cv2
import glob
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from PIL import Image
import albumentations as A#数据增强库

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


# transform = A.Compose([
#     A.RandomRotate90(p=0.5),  # 旋转
#     A.HorizontalFlip(p=0.5),
#     # A.ShiftScaleRotate(p=0.1),#
#     A.MedianBlur(blur_limit=[1, 7], p=0.1),
#     A.MotionBlur(blur_limit=[5, 12], p=0.1),
#     A.RGBShift(p=0.05),
#     A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1),
# ])

transform = A.Compose([
    A.RandomRotate90(p=0.5),  # 旋转
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.1), # 随机仿射变换

    # A.RandomGamma(p=0.1), # 亮度增强
    A.MedianBlur(blur_limit=[1, 7], p=0.1),
    A.MotionBlur(blur_limit=[5, 12], p=0.1),
    A.RGBShift(p=0.05),
    A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1),
])

TFS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    #transforms.RandomCrop((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class Dataset(Dataset):
    def __init__(self, root_dir, data_list, local_rank, transform=TFS,augment=False):
        super(Dataset, self).__init__()
        self.augment=augment # 是否数据增强
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        with open(os.path.join(data_list), 'r') as fd:
            imgs = fd.readlines()
        self.imgs = [] 
        self.weights = []
        for img in imgs:
            data = os.path.join(root_dir, img[:-1])
            self.imgs.append(data)
            w = float(data.split(' ')[-1])
            self.weights.append(w)

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = cv2.imread(img_path)
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)

        if self.augment: #如果使用数据增强
            transformed = transform(image=data) #增强
            data = Image.fromarray(transformed["image"])
        data = self.transform(data)
        label = np.int32(splits[1])
        return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = Dataset(root_dir='', data_list='ttttt.txt', local_rank=0)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, len(dataset.weights))
    trainloader = DataLoaderX(local_rank=0, dataset=dataset, batch_size=4, sampler=sampler)
    for i, (data, label) in enumerate(trainloader):
        print(data.shape, label)
        #img = torchvision.utils.make_grid(data).numpy()
        #img = np.transpose(img, (1, 2, 0))
        #img += np.array([1, 1, 1])
        #img *= 127.5
        #img = img.astype(np.uint8)
        #img = img[:, :, [2, 1, 0]]

        #cv2.imshow('img', img)
        #cv2.waitKey()

