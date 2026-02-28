import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ===================== 1. 配置环境与超参数 =====================
# 自动选择设备：优先用MPS（苹果M芯片）→ 其次CUDA → 最后CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数（新手可直接用，无需修改）
BATCH_SIZE = 64    # 每次训练的样本数
LEARNING_RATE = 0.001  # 学习率
EPOCHS = 5         # 训练轮数

# ===================== 2. 数据加载（MNIST手写数字数据集） =====================
# 数据预处理：转为张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 把图片转为张量（0-1）
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值/方差（固定值）
])

# 加载训练集和测试集（PyTorch内置，自动下载）
train_dataset = datasets.MNIST(
    root="./data",  # 数据保存路径
    train=True,     # 训练集
    download=True,  # 自动下载（首次运行需要）
    transform=transform
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,    # 测试集
    transform=transform
)

# 数据加载器（批量加载+打乱数据）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 3. 定义简单的全连接神经网络（MLP） =====================
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

# 初始化模型并移到指定设备（MPS/CPU/CUDA）
model = SimpleMNISTModel().to(device)

# ===================== 4. 定义损失函数和优化器 =====================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务专用）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器（常用且稳定）

# ===================== 5. 训练模型 =====================
def train_model():
    model.train()  # 切换到训练模式
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到指定设备
        data, target = data.to(device), target.to(device)
        
        # 清零梯度（避免累积）
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播（求梯度）
        loss.backward()
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每100个批次打印一次进度
        if batch_idx % 100 == 0:
            print(f"训练轮数: {epoch+1}/{EPOCHS} | 批次: {batch_idx}/{len(train_loader)} | 损失: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    train_time = time.time() - start_time
    print(f"训练轮数 {epoch+1} 完成 | 平均损失: {avg_loss:.4f} | 耗时: {train_time:.2f}秒")

# ===================== 6. 验证模型（测试集） =====================
def test_model():
    model.eval()  # 切换到验证模式（禁用Dropout等）
    test_loss = 0.0
    correct = 0
    
    # 验证时禁用梯度计算（节省资源）
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 累加损失
            test_loss += criterion(output, target).item()
            # 计算预测结果（取概率最大的类别）
            pred = output.argmax(dim=1, keepdim=True)
            # 统计正确数
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # 计算指标
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_dataset)
    print(f"验证结果 | 平均损失: {test_loss:.4f} | 准确率: {correct}/{len(test_dataset)} ({accuracy:.2f}%)\n")

# ===================== 7. 主训练循环 =====================
if __name__ == "__main__":
    print("开始训练简单MNIST分类模型...")
    for epoch in range(EPOCHS):
        train_model()
        test_model()
    
    # 保存训练好的模型（可选）
    torch.save(model.state_dict(), "simple_mnist_model.pth")
    print("模型训练完成，已保存为 simple_mnist_model.pth")