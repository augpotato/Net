import torch
from debugpy.launcher import channel
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):  #卷积板块
    def __init__(self,in_channel,out_channel):
        super(Conv_Block,self).__init__()  #初始化父类方法，固定格式
        self.layer = nn.Sequential(
            # Conv2d卷积格式(输入，输出，卷积核大小，padding,步长，padding格式，bias)
            # 以下为第一次卷积
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 卷积层后用来进行数据的归一化处理，使得数据在进行Relu之前不会因为数据过大而不稳定
            nn.Dropout2d(0.3),  # 在前向传播时让某个神经元的激活值以一定的概率P停止工作，使模型泛化更强，不依赖局部的特征
            nn.LeakyReLU(),  # 激活

            # 以下为第二次卷积,把第一次卷积的out作为输入
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self,x): #前向传播
        return self.layer(x)

#下采样实现
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

#上采样实现,采用插值法实现,不用转置卷积因为会产生一圈空洞，对分割影响大
#根据U-Net最下方，先上采样，再与之前的图进行concat拼接，再进行卷积
class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1) #上采样后的1x1卷积,通道除以二

    def forward(self, x, feature_map): #feature是之前的特征图，用于concat拼接
        up = F.interpolate(x, scale_factor=2, mode='nearest') #插值法
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1) #N,C,H,W,在C通道上进行的，故dim=1,进行concat拼接

class UNet(nn.Module):  #按照算法逻辑图去实现
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)  #卷积输入三个通道，输出的是64
        self.d1 = DownSample(64)  #下采样输入64
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)  #开始进行上采样
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1) #输出三通道
        self.Th = nn.Sigmoid() #激活

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1 = self.c6(self.u1(R5, R4)) #上采样，后进行拼接，再卷积
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)