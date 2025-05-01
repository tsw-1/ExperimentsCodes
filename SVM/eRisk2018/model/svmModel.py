import numpy as np
import json
from sklearn.metrics import f1_score,precision_score, recall_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import json
from scipy.sparse import csr_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer


'''根据SVM对测试集进行预测'''

#   训练集及标签
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/data3.json','r')as f:
    data=json.load(f)
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/average_features4_2.json','r')as f:
    average_features=np.array(json.load(f))    
train=data['train']
for i in range(len(train)):
    train[i].extend(average_features[i])
X=np.array(train)
y=np.array((data['y']))

# 找出元素全为0的行的索引
zero_row_indices = np.where(np.all(X == 0, axis=1))[0]
# 移除全为0的行
X = np.delete(X, zero_row_indices, axis=0)
y = np.delete(y, zero_row_indices, axis=0)
X=csr_matrix(X)

#   测试集及标签
test=data['X_test']
for i in range(len(test)):
    test[i].extend(average_features[887+i])
X_test=np.array(test)
X_test=csr_matrix(X_test)
y_test=np.array(data['y_test'])
# # 创建SMOTE对象,过采样
# sm=SMOTE(sampling_strategy='auto')
# X,y=sm.fit_resample(X,y)

# 动态随机欠采样
rus = RandomUnderSampler(sampling_strategy='auto',random_state=100, replacement=False)
X, y = rus.fit_resample(X, y)

# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

 # 定义SVM分类器
# clf = svm.LinearSVC(penalty='l2', dual=False, C=C,class_weight=class_weights)
# clf = svm.LinearSVC(C=0.12,class_weight=class_weights)
list=[]
for C in np.arange(0.01,0.5,0.01):
    clf = SVC(C=C, class_weight=class_weights, probability=True)
    dict={}
    # 训练模型
    clf.fit(X, y)                                                                   
    # 测试模型
    predicted = clf.predict(X_test)
 
    # 预测概率值
    probabilities = clf.predict_proba(X_test)

    accuracy = 100 * np.sum(predicted == y_test) / len(y_test)
    f1 = f1_score(y_test, predicted)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)

    print('Accuracy of the model on the test samples: {:.2f} %'.format(accuracy))
    print('F1 score of the model on the test samples: {:.2f}'.format(f1))
    print("================================================")
    dict['f1']=f1
    dict['acc']=accuracy
    dict['c']=C
    dict['p']=precision
    dict['r']=recall
    list.append(dict)
print("================================================")



#%%
#  测试集标签
import json
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/data3.json','r')as f:
    data=json.load(f)
with open ('/home/E22301339/depressionDetection-eRisk2018/features/test-golden-truth-test.txt','r+')as f:
        list_data=[]
        list_total=[]  # 存储positive抑郁的用户id
        for item in f.readlines():
            list_data.append(item.strip())
str=list_data[0][:-1].strip()

for item in list_data:
    if(item[len(item)-1]=='1'):
        list_total.append(item[:-1].strip())
    else:
        break
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/total_user_wordpost2.json','r')as f:
        users=list(json.load(f).keys())
labels=[]
for user in users:
    if user in list_total:
        labels.append(1)
    else:
        labels.append(0)
test_y=labels
data['y_test']=test_y
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/data3.json','w')as f:
    json.dump(data,f)

print("Process")

# %%
