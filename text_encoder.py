# 该模块主要是为了实现text编码，但是因为只有十类，所以值需要用nn.emdding进行初始化就行

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn

'''
# Part2 设计下文本编码器的类
'''
class TextEncoder(nn.Module):
    def __init__(self,voca_size,emd_size=16,f_size=64,final_emd_size=8):
        super().__init__()

        # 首先需要初始化嵌入的类别和维度
        self.emd=nn.Embedding(num_embeddings=voca_size,embedding_dim=emd_size)
        # 对嵌入的维度进行初始化
        self.linear1=nn.Linear(emd_size,f_size)
        self.linear2=nn.Linear(f_size,emd_size)
        self.linear3=nn.Linear(emd_size,final_emd_size)

        self.ln=nn.LayerNorm(final_emd_size)

    def forward(self,batch_label):
        batch_label_emd=self.emd(batch_label)
        batch_label_emd=self.linear1(batch_label_emd)
        batch_label_emd=self.linear2(batch_label_emd)
        batch_label_emd=self.linear3(batch_label_emd)
        return self.ln(batch_label_emd)

'''
# 测试
'''
if __name__=='__main__':
    text_encoder=TextEncoder(voca_size=10,emd_size=16,f_size=64,final_emd_size=8)
    x=torch.tensor([1,2,3,4,5,6,7,8,9,0])
    y=text_encoder(x)
    print(y.shape)