#对图片数据集进行基本的处理与测试
import os.path

import torch.cuda
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms


transform=transforms.Compose([  #图片归一化
    transforms.ToTensor()
])

class Mydataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):  #文件名的 数量
        return len(self.name)

    def __getitem__(self, index): #数据追踪
        segment_name=self.name[index]  #对应下标文件的名字xx.png
        segment_path=os.path.join(self.path, 'SegmentationClass', segment_name) #地址拼接
        image_path=os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))#拼接与后缀修改
        segment_image=keep_image_size_open(segment_path) #调整图片尺寸
        image=keep_image_size_open(image_path)
        return transform(image),transform(segment_image)

if __name__ =='__main__':
    data = Mydataset('D:\\数据集\\UNET\\VOCdevkit\\VOC2012')   #数据集文件夹
    print(data[0][0].shape)
    print(torch.cuda.is_available())

