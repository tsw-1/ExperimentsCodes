import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import f1_score
from sklearn import svm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
    
'''根据SVM、LR对测试集进行预测'''


num_samples = 806
num_features = 1000

with open('/home/E22301339/depressionDetection/processedData/data.json','r')as f:
    data2=json.load(f)

#   训练集及标签
with open('/home/E22301339/depressionDetection/processedData/user_features_train.json','r')as f:
    data=json.load(f)
data=list(data.values())
y=data2['y']

# 创建SMOTE对象,过采样
# sm=SMOTE(sampling_strategy='auto')
# data,y=sm.fit_resample(data,y)

# # 下采样
# data, y = RandomUnderSampler().fit_resample(data, y)

#   测试集及标签
with open('/home/E22301339/depressionDetection/processedData/user_features_test.json','r')as f:
    test_data=json.load(f)
test_data=list(test_data.values())
test_y=data2['y_test']


X=np.array(list(data))
y=np.array(y)


# 将数据转换为PyTorch张量
x_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).long()

# 定义SVM分类器
clf = svm.SVC()

# 训练模型
clf.fit(X, y)

# 测试模型
X_test = np.array(test_data)
y_test = np.array(test_y)
predicted = clf.predict(X_test)
accuracy = 100 * np.sum(predicted == y_test) / len(y_test)
f1 = f1_score(y_test, predicted)
print('Accuracy of the model on the test samples: {:.2f} %'.format(accuracy))
print('F1 score of the model on the test samples: {:.2f}'.format(f1))

















