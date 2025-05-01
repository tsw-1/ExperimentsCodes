from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import json

X,y=make_classification(n_classes=2,weights=[0.1,0.9],n_features=20,n_samples=5000)

# 创建SMOTE对象,过采样
sm=SMOTE(sampling_strategy='auto')
# X_resampled,y_resampled=sm.fit_resample(X,y)
# print("===========")

with open('/home/E22301339/depressionDetection/processedData/data.json','r')as f:
    data=json.load(f)
train=data['train']
y_train=data['y']

X_resampled,y_resampled=sm.fit_resample(train,y_train)
print("===========")
