import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 数据准备
# 假设我们有一个包含两个特征的数据集，每个特征都是连续值
# 我们使用随机生成的数据作为示例
num_samples = 1000
num_features = 2
X = np.random.randn(num_samples, num_features)
# 假设标签为0或1
y = np.random.randint(2, size=num_samples)

# 将数据转换为PyTorch张量
x_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).long()

# 定义模型
class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 2)

    def forward(self, x):
        x = self.linear(x)
        return x

model = LogisticRegression(num_features)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
with torch.no_grad():
    num_test_samples = 100
    X_test = np.random.randn(num_test_samples, num_features)
    y_test = np.random.randint(2, size=num_test_samples)
    x_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    outputs = model(x_test)
    _, predicted = torch.max(outputs.data, 1)
    print('Accuracy of the model on the test samples: {} %'.format(100 * torch.sum(predicted == y_test) / num_test_samples))
