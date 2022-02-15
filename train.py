from torch import nn, optim
from torch.utils.data import DataLoader

from Data import *
from net import *
from torchvision.utils import save_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'   #存储最后更新的权重的地方
data_path = r'D:\数据集\UNET\VOCdevkit\VOC2012'
save_path = 'trained_image'


if __name__ == '__main__':
    data_loader = DataLoader(Mydataset(data_path), batch_size=2, shuffle=True)  #batch_size为2意思是每个batch两个样本，数据总量除以2就是batch数
    net = UNet().to(device)   #模型放入训练
    if os.path.exists(weight_path):    #加载卷重
        net.load_state_dict(torch.load(weight_path))
        print('成功加载权重')
    else:
        print('加载权重失败')

    opt = optim.Adam(net.parameters())    #优化权重衰减
    loss_fun = nn.BCELoss()  #对一个batch里面的数据做二元交叉熵并且求平均。求损失函数专用

    epoch = 1   #训练轮次
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)   #这里要放入训练的图片都要加上.to(device)转为GPU格式训练，否则为CPU格式训练

            out_image = net(image)     #输出的图像为经过U-net网路的图
            train_loss = loss_fun(out_image, segment_image)   #损失函数计算

            opt.zero_grad()    #清空梯度，防止显存压力大
            train_loss.backward()  #计算损失函数梯度
            opt.step()  #更新所有参数

            if i%5==0:
                print(f'{epoch}-{i}-train_loss=====>>{train_loss.item()}')

            if i%50==0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)   #图像拼接，原图、标签图、结果图
            save_image(img, f'{save_path}/{i}.png')   #保存结果

        epoch+=1