import json
import numpy as np
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# 读取训练集及标签
with open('/home/E22301339/depressionDetection/processedData/data4.json','r')as f:
    data=json.load(f)
with open('/home/E22301339/depressionDetection/processedData/average_features2.json','r')as f:
    average_features=np.array(json.load(f))

train = data['train']
for i in range(len(train)):
    train[i].extend(average_features[i])

X_train = np.array(train)
y_train = np.array(data['y'])

# 读取测试集及标签
test=data['X_test']
for i in range(len(test)):
    test[i].extend(average_features[486+i])

X_test = np.array(test)
y_test = np.array(data['y_test'])

# 合并训练集和测试集
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# 找出元素全为0的行的索引并移除
zero_row_indices = np.where(np.all(X_combined == 0, axis=1))[0]
X_combined = np.delete(X_combined, zero_row_indices, axis=0)
y_combined = np.delete(y_combined, zero_row_indices, axis=0)
X_combined = csr_matrix(X_combined)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300000, random_state=0, init="random")
X_tsne = tsne.fit_transform(X_combined)

# 将数据点按标签分组
X_class0 = X_tsne[y_combined == 0]
X_class1 = X_tsne[y_combined == 1]

# 分配颜色
color_class0 = 'blue'
color_class1 = 'red'

# 绘制散点图
plt.scatter(X_class0[:, 0], X_class0[:, 1], c=color_class0, label='Control')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c=color_class1, label='Depressed')

# 添加图例
plt.legend()

# 显示图像
plt.show()
plt.savefig('tsne_plot2017_7.png')  # 保存图像到文件