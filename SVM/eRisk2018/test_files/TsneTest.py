import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 生成模拟数据
np.random.seed(0)
n_samples = 100
n_features = 10
X = np.random.rand(n_samples, n_features)
labels = np.random.randint(0, 2, n_samples)  # 生成随机标签，这里假设有两个类别

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
X_tsne = tsne.fit_transform(X)

# 将数据点按标签分组
X_class0 = X_tsne[labels == 0]
X_class1 = X_tsne[labels == 1]

# 分配颜色
color_class0 = 'blue'
color_class1 = 'red'

# 绘制散点图
plt.scatter(X_class0[:, 0], X_class0[:, 1], c=color_class0, label='Class 0')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c=color_class1, label='Class 1')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 添加图例
plt.legend()

# 显示图像
plt.show()
plt.savefig('tsne_plot2.png')  # 保存图像到文件
