import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
import pandas as pd

'''代码块作用：通过td-idf以及卡方分布进行特征提取，提取和抑郁症最相关的的子情绪序列'''


#  读取json数据
with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost1.json', 'r') as f:
    data = json.load(f)
with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost2.json', 'r') as f:
    data2 = json.load(f)
with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost3.json', 'r') as f:
    data3 = json.load(f)
with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost4.json', 'r') as f:
    data4 = json.load(f)
with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost5.json', 'r') as f:
    data5 = json.load(f)
data.update(data2)  # dict追加
data.update(data3)
data.update(data4)
data.update(data5)

# data={"train_subject64": [
#         "positive235",
#         "positive61",
#         "surprise55",
#         "trust121",
#         "sadness192"
# ],"train_subject71": [
#         "positive206",
#         "positive3",
#         "sadness192",
#         "positive235",
#         "positive266"
        
# ]
# }

# 计算每个文档的tf-idf权重矩阵
vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform([' '.join(doc) for doc in data.values()])

# 创建字典来存储每个文档的tf-idf值
tf_idf_dict = {}

# 统计每个子情绪的tf-idf值
for name, doc in data.items():
    # 将当前文档转换为tf-idf权重矩阵
    doc_tf_idf = vectorizer.transform([' '.join(doc)])
    doc=set(doc)
    # 获取当前文档中每个单词的tf-idf值
    word_tf_idf = []
    for word in doc:
        # 获取单词在词汇表中的索引
        word_index = vectorizer.vocabulary_.get(word)
        if word_index is not None:
            # 计算单词的tf-idf值
            tf_idf_value = doc_tf_idf[0, word_index]
            word_tf_idf.append((word, tf_idf_value))
    
    # 将当前文档中每个单词的tf-idf值按照值进行降序排列
    word_tf_idf = sorted(word_tf_idf, key=lambda x: x[1], reverse=True)
    
    # 将当前文档的tf-idf值存储在字典中
    tf_idf_dict[name] = word_tf_idf

# 将当前文档的tf-idf值存储在字典中
tf_idf_dict[name] = word_tf_idf

# 取每个用户排名前1000的sub-emotion，存储在一个列表中，并将该列表与用户名称一起存储在一个字典中。
top_sub_emotions = {name: [word[0] for word in tf_idf_dict[name]] for name in tf_idf_dict}

# 处理成字典
for key in tf_idf_dict:
    temp=tf_idf_dict[key]
    dict_temp={}
    for item in temp:
        dict_temp[item[0]]=item[1]
    tf_idf_dict[key]=dict_temp

with open('/home/E22301339/depressionDetection/features/train_features.json','w')as f:
    json.dump(tf_idf_dict, f)

# 将所有用户的 top_sub_emotions 列表取并集，作为初始特征集
union_sub_emotions = list(set().union(*top_sub_emotions.values()))


# 对每个用户进行编码
features = np.zeros((len(data), len(union_sub_emotions)))
for i, (name, sub_emotions) in enumerate(top_sub_emotions.items()):
    for j, sub_emotion in enumerate(union_sub_emotions):
        if sub_emotion in sub_emotions:
            features[i, j] =  tf_idf_dict[name][sub_emotion]



# 存储每个用户的one-hot编码特征
user_features = {}
for i, name in enumerate(data.keys()):
    user_features[name] = list(features[i])

with open('/home/E22301339/depressionDetection/features/user_features_train.json','w')as f:
    json.dump(user_features, f)


'''
list_total存储用户抑郁症标签
'''
with open ('/home/E22301339/depressionDetection/features/risk_golden_truth.txt','r+')as f:
   list_data=[]
   list_total=[]  # 存储positive抑郁的用户id
   for item in f.readlines():
      list_data.append(item.strip())
str=list_data[0][:-1].strip()
print(str)
print(list_data[0][len(list_data[0])-1])

for item in list_data:
   if(item[len(item)-1]=='1'):
      list_total.append(item[:-1].strip())
   else:
      break


 
'''卡方分布提取特征''' 

# 输入特征
X = np.array(list(user_features.values()))
# 抑郁症标签
y = []
for id in user_features.keys():
    if id in list_total:
        y.append(1)
    else:
        y.append(0)
y=np.array(y)

# 设置要选择的特征数量
num_features = 1000

# # 计算每个特征与目标变量之间的卡方值和p值
# for i in range(X.shape[1]):
#     contingency_table = pd.crosstab(X[:,i], y)
#     chi2, p, dof, expected = chi2_contingency(contingency_table)
#     print(f"Feature {i}: chi2={chi2}, p={p}")           

# # 根据设定的显著性水平，选择卡方值和p值都超过该水平的特征
# significant_features = {}
# alpha = 0.05
# for i in range(X.shape[1]):
#     contingency_table = pd.crosstab(X[:,i], y)
#     chi2, p, dof, expected = chi2_contingency(contingency_table)
#     if p < alpha and chi2 > stats.chi2.ppf(q=1-alpha, df=dof):
#         significant_features[i]=union_sub_emotions[i]

# print(f"Significant features: {significant_features}")

# 使用SelectKBest进行特征选择
selector = SelectKBest(chi2, k=num_features)
X_selected = selector.fit_transform(X, y)

# 获取选择的特征索引
selected_indices = selector.get_support(indices=True)

# 获取选择的特征名称
selected_features = [union_sub_emotions[i] for i in selected_indices]

print(f"Selected features: {selected_features}")



with open('/home/E22301339/depressionDetection/features/significant_features.json','w')as f:
    json.dump(selected_features, f)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       














