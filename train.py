# 定义模型的训练
'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
from dataset import Mnist_dataset
from torch.utils import data
from CLIP import CLIP

'''
# Part2 初始化一些训练的参数
'''

if __name__ == '__main__':
    # 初始化数据迭代器
    batch_size = 64
    Mnist_dataloader = data.DataLoader(Mnist_dataset, batch_size=batch_size, shuffle=True)
    # 一些超参数的设置
    epochs = 100
    lr = 1e-2
    # 初始化前向模型
    clip = CLIP(in_channnel=1, voca_size=10, emd_size=8)

    # 定义损失函数
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    # 定义优化器
    optim = torch.optim.Adam(params=clip.parameters(), lr=lr)

    # 开始训练
    num_kind = 10
    for epoch in range(epochs):
        for batch_img, batch_label in Mnist_dataloader:
            # 得到的一个batch_size的数据。需要从里面挑出对应的0-9，因为对称矩阵需要对象之间是没关系的
            # 首先如果里面就不包含10类，得跳过这个batch_size
            if torch.unique(batch_label).size()[0] < num_kind:
                continue
            # 如果有包含10类的话
            is_in = torch.zeros(size=(10,))
            target_index = []
            for i, item in enumerate(batch_label):
                # 对应的元素不在里面
                if is_in[item] == 0:
                    target_index.append(i)
                    is_in[item] = 1
                    if len(target_index) == num_kind:
                        break
                else:
                    continue
            # 此时就得到的batch_size里面的所有的对应的10个index
            batch_img_select = batch_img[target_index]
            batch_label_select = batch_label[target_index]
            # 计算对应的loss
            pre = clip(batch_img_select, batch_label_select)
            l1 = loss1(pre, torch.eye(n=pre.size()[0], m=pre.size()[1]))
            l2 = loss2(pre.transpose(0, 1), torch.eye(n=pre.size()[0], m=pre.size()[1]))
            l = (l1 + l2) / 2
            # 初始化梯度
            optim.zero_grad()
            # 计算梯度
            l.backward()
            # 参数进行更新
            optim.step()
            # 得到对应损失
            loss_last = l.item()
        print('此时的epoch为{}，loss为{}'.format(epoch, loss_last))
