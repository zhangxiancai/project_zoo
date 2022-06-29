import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import cv2
import os

class FireDataset(Dataset): #单个载入
    def __init__(self,data_root,train_val_address):
        super(FireDataset, self).__init__()
        self.data_root=data_root
        with open(os.path.join(self.data_root,train_val_address)) as f:
            imgs_labs=f.readlines()
        self.imgs_labs=imgs_labs
        self.transform=transform

    def __getitem__(self, index):#
        line=self.imgs_labs[index].strip()
        ad,lab=line.split(' ')
        f=os.path.join(self.data_root, ad)
        img=cv2.imread(f) #HWC
        img=self.transform(img) #CHW
        return img, lab

    def __len__(self):
        return len(self.imgs_labs)

# class ToTensor(object):
#     def __call__(self, img):
#         return torch.from_numpy(img)

transform=transforms.Compose([transforms.ToTensor()
                    ,transforms.Resize((112,112))])

# ,transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
# class FireDataloader(DataLoader):#batch载入，shuffle，多线程载入


if __name__=='__main__':
    dataset=FireDataset('/home/xiancai/Ruler/pytorch-image-classfication/data','train.txt')
    dataloader=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=2)
    for i, (img,lab) in enumerate(dataset):
        if i>3:
            break
        print(f'index:{i},lab:{lab}')
        print(img.shape)
        img=img.permute(1,2,0)
        print(img.shape)
        cv2.imwrite(f'{i}_lab{lab}.jpg',img.numpy()*255)



