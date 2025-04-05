# 该模块主要是为了实现CLIP模型，输入图像和文本得到对应的嵌入向量
'''
# Part1引入相关的库函数
'''
import torch
from torch import nn
from imag_encoder import ImagEncoder
from text_encoder import TextEncoder

'''
# Part2 构建CLIP类函数
'''


class CLIP(nn.Module):
    def __init__(self, in_channnel, voca_size, emd_size):
        super().__init__()
        self.img_encoder = ImagEncoder(in_channel=in_channnel, final_emd=emd_size)
        self.text_encoder = TextEncoder(voca_size=voca_size, final_emd_size=emd_size)

    def forward(self, batch_img, batch_text):
        batch_img_emd = self.img_encoder(batch_img)  # (batch,emd_size)
        batch_text_emd = self.text_encoder(batch_text).transpose(0, 1)  # (batch,emd_size)
        return torch.matmul(batch_img_emd, batch_text_emd)


'''
# Part3 测试
'''

if __name__ == '__main__':
    clip = CLIP(emd_size=8, in_channnel=1, voca_size=10)
    img_x = torch.randn(5, 1, 28, 28)
    text_x = torch.randint(0, 10, (5,))
    logits = clip(img_x, text_x)
    print(logits.shape)
