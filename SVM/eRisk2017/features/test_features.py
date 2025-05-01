#%%
'''代码块作用：根据测试集初始特征集和抑郁症最相关的子情绪序列来对用户进行编码'''

import json
with open ('/home/E22301339/depressionDetection/features/significant_features.json','r')as f:
    significant_features=json.load(f)
significant_features=list(significant_features)

with open ('/home/E22301339/depressionDetection/features/train_features.json','r')as f:
    test_features=json.load(f)

user_features_test={}
for test_feature in test_features:
    feature_list=[]
    features=test_features[test_feature]
    for i in range(len(significant_features)):
        if significant_features[i] in features.keys():
            feature_list.append(features[significant_features[i]])
        else:
            feature_list.append(0)
    user_features_test[test_feature]=feature_list

with open('/home/E22301339/depressionDetection/features/user_features_train.json','w')as f:
    json.dump(user_features_test, f)

print("================================================")

#%%

'''代码块作用：根据list_total来对测试集用户抑郁症标签进行编码'''

import json
with open ('/home/E22301339/depressionDetection/features/test_golden_truth.txt','r+')as f:
        list_data=[]
        list_total=[]  # 存储positive抑郁的用户id
        for item in f.readlines():
            list_data.append(item.strip())
str=list_data[0][:-1].strip()
    # print(str)
    # print(list_data[0][len(list_data[0])-1])

for item in list_data:
    if(item[len(item)-1]=='1'):
        list_total.append(item[:-1].strip())
    else:
        break
with open('/home/E22301339/depressionDetection/features/user_features_test.json','r')as f:
        users=list(json.load(f).keys())

labels=[]

for user in users:
    if user in list_total:
        labels.append(1)
    else:
        labels.append(0)
print (labels)
# %%
