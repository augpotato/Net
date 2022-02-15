import os.path

import torch

from net import *
from utils import *
from Data import *
from torchvision.utils import save_image

net = UNet().cuda()

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('权重加载成功')
else:
    print('权重加载失败')


_input = input('请放入要分割的图片:')  #在test_image里放图片，用绝对路径

img = keep_image_size_open(_input)  #格式化图片
img_data = transform(img).cuda()   #将归一化后的图片放入GPU开测
print(img_data.shape)
img_data = torch.unsqueeze(img_data, dim=0)
out = net(img_data)
save_image(out, 'result/result.jpg')  #训练后的结果放在result里面
print(out)

