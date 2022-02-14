#工具类，对图片大小尺寸进行调整
from PIL import Image

#使数据集每一张图片大小一致
def keep_image_size_open(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size) #取图片最长边
    mask=Image.new('RGB',(temp,temp),(0,0,0)) #图片掩码，RGB通道、最长边正方形，原色
    mask.paste(img,(0,0)) #加入图片
    mask=mask.resize(size)
    return mask