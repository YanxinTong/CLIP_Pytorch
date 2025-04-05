# 该模块主要为了实现CLIP的Mnist数据集的实现
'''
# Part1 引入相关的库函数
'''
import torch
from torch.utils import data
import torchvision
from torchvision import transforms

'''
# Part2 实现数据的预处理和数据集的下载
'''

transform_action=transforms.Compose([
    transforms.ToTensor() # 从Pillow到Tensor，除了255，变换了通道的顺序(img_size,img_size,channel)->(channel,img_size,img_size)
])

Mnist_dataset=torchvision.datasets.MNIST(root='Mnist_dataset',train=True,transform=transform_action,download=True)

'''
# Part3 测试
'''

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = Mnist_dataset
    img, label = ds[0]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
