# 该模块主要是为了实现图像的嵌入，把一个图像(batch,channel,img_size,img_size)->变成(batch,emd_size)
'''
# Part1 引入相关库函数
'''
import torch
from torch import nn

'''
# Part2 初始化类函数
'''


# 首先先定义一个残差卷积的函数
class ResdualModule(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        # 首先是自己的前向过程
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1,
                               stride=stride)  # 变化和conv3统一
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1,
                               stride=1)  # 使得面积不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        # 然后是1*1的卷积前向残差过程
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1), padding=0,
                               stride=stride)  # 和第一次变化一样

        # 相加之后的激活
        self.relu1 = nn.ReLU()

    def forward(self, x):  # (batch,1,28,28) # 当stride=1的时候
        # 首先是自己的前向过程
        x1 = self.conv1(x)  # (batch,1,28,28)
        x1 = self.bn1(x1)  # (batch,1,28,28)
        x1 = self.relu(x1)  # (batch,1,28,28)
        x1 = self.conv2(x1)  # (batch,1,28,28)
        x1 = self.bn2(x1)  # (batch,1,28,28)
        # 然后是残差的前向过程
        x2 = self.conv3(x)  # (batch,1,28,28)
        return self.relu1(x1 + x2)  # (batch,1,28,28)


# 然后在残差的基础上构建图像编码器，最终要把一个图像变成和文本编码同长度的向量
class ImagEncoder(nn.Module):
    def __init__(self, in_channel, final_emd, f1_channel=16, f2_channel=4, out_channel=1, stride1=2, stride2=2,
                 stride3=2, f_feature=16):
        super().__init__()
        self.res1 = ResdualModule(in_channel=in_channel, out_channel=f1_channel,
                                  stride=stride1)  # (batch,f1_channel,(img_size-3)+p*2//stride)
        self.res2 = ResdualModule(in_channel=f1_channel, out_channel=f2_channel, stride=stride2)
        self.res3 = ResdualModule(in_channel=f2_channel, out_channel=out_channel, stride=stride3)
        self.linear1 = nn.Linear(in_features=f_feature,
                                 out_features=final_emd)  # 这个因为如果不能整除的化，可能会出现上下取整，那么输入的feature就不能确定
        self.ln = nn.LayerNorm(final_emd)

    def forward(self, x):
        x = self.res1(x)  # (batch,16,14,14)
        x = self.res2(x)  # (batch,4,7,7)
        x = self.res3(x)  # (batch,1,4,4)
        x = x.reshape(x.size()[0], -1)  # batch,16
        x = self.linear1(x)  # batch,8
        x = self.ln(x)  # batch,8
        return x


'''
# Part3 开始测试
'''
if __name__ == '__main__':
    img_encoder = ImagEncoder(in_channel=1, final_emd=8)
    out = img_encoder(torch.randn(1, 1, 28, 28))
    print(out.shape)
