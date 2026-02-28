import torch
import torch.nn as nn

class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        # 模型结构：输入（28×28=784）→ 隐藏层128 → 隐藏层64 → 输出10（0-9数字）
        self.flatten = nn.Flatten()  # 把28×28的图片展平为784维向量
        self.fc1 = nn.Linear(784, 128)  # 全连接层1：784→128
        self.relu = nn.ReLU()          # 激活函数（增加非线性）
        self.fc2 = nn.Linear(128, 64)   # 全连接层2：128→64
        self.fc3 = nn.Linear(64, 10)    # 输出层：64→10（对应10个数字类别）

    def forward(self, x):
        # 前向传播（模型的核心计算逻辑）
        x = self.flatten(x)    # [batch, 1, 28, 28] → [batch, 784]
        x = self.fc1(x)        # [batch, 784] → [batch, 128]
        x = self.relu(x)       # 激活
        x = self.fc2(x)        # [batch, 128] → [batch, 64]
        x = self.relu(x)       # 激活
        x = self.fc3(x)        # [batch, 64] → [batch, 10]
        return x
