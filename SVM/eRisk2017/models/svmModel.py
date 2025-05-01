import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义 SVM 模型
class SVM(nn.Module):
    def __init__(self, num_features):
        super(SVM, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

# 生成训练数据
num_samples = 100
num_features = 10
X = np.random.randn(num_samples, num_features)
Y = np.random.randint(0, 2, size=(num_samples, 1))

# 转换为张量
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# 定义模型和优化器
model = SVM(num_features)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = torch.mean(torch.clamp(1 - Y * model(X), min=0))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
with torch.no_grad():
    predicted = model(X).sign()
    accuracy = (predicted == Y).float().mean()
    print('Accuracy: {:.4f}'.format(accuracy.item()))
