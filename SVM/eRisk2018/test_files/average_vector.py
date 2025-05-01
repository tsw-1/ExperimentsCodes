# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
import json
import numpy as np

# %%
# 平均向量作特征(word)
model = fasttext.load_model('/home/E22301339/fastText/fastText/cc.en.300.bin')
with open('/home/E22301339/depressionDetection/processedData/total_user_wordpost.json', 'r')as f:
    words = json.load(f)
with open('/home/E22301339/depressionDetection/processedData/total_user_wordpost2.json', 'r')as f:
    words2 = json.load(f)
words = list(words.values())
words2 = list(words2.values())
words.extend(words2)
temp = []
for item in words:
    vectors = []
    for word in item:
        vector = model[word]
        vectors.append(vector)
    if len(vectors) > 0:
        average_vector = np.mean(vectors, axis=0)
        temp.append(average_vector.tolist())
    else:
        # 处理词向量缺失的情况
        print("存在nan")
        temp.append([0] * model.get_dimension())
with open('/home/E22301339/depressionDetection/processedData/average_features.json', 'w')as f:
    json.dump(temp, f)
print("================================")

# %%
# 平均向量作特征(sub-emotion)
with open('/home/E22301339/depressionDetection/codes-author/data/train2.json', 'r')as f:
    sub_emotions = json.load(f)
with open('/home/E22301339/depressionDetection/codes-author/data/test2.json', 'r')as f:
    sub_emotions2 = json.load(f)
with open('/home/E22301339/depressionDetection/codes-author/emotions_ext/sub-emotions_2.0.json', 'r')as f:
    trust_cluster_vectos = json.load(f)
sub_emotions = list(sub_emotions.values())
sub_emotions2 = list(sub_emotions2.values())
sub_emotions.extend(sub_emotions2)
temp = []
for item in sub_emotions:
    vectors = []
    for sub_emotion in item:
        vector = trust_cluster_vectos[sub_emotion]
        vectors.append(vector)
    if len(vectors) > 0:
        average_vector = np.mean(vectors, axis=0)
        temp.append(average_vector.tolist())
    else:
        # 处理词向量缺失的情况
        print("存在nan")
        temp.append([0] * 300)
with open('/home/E22301339/depressionDetection/processedData/average_features2.json', 'w')as f:
    json.dump(temp, f)
print("================================")

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
#   TFIDF平均加权词向量作特征
tfidVector = TfidfVectorizer(smooth_idf=False,sublinear_tf=True)
tfidVector2 = TfidfVectorizer(smooth_idf=False)
#  读取子情绪序列数据
with open('/home/E22301339/depressionDetection/codes-author/data/train2.json', 'r') as f:
    data = json.load(f)
data = list(data.values())
for i in range(len(data)):
    data[i] = ' '.join(data[i]).strip()
train = data
text = train

with open('/home/E22301339/depressionDetection/codes-author/data/test2.json', 'r') as f:
    data = json.load(f)
data = list(data.values())
for i in range(len(data)):
    data[i] = ' '.join(data[i]).strip()
test = data
text2 = test
# text.extend(text2)
# text = ['anger1 positive1 anger2 negative4 negative4', 'anger3 joy1 joy2']

tfidf_matrix = tfidVector.fit_transform(text)
tfidf_matrix2 = tfidVector2.fit_transform(text2)
print("使用tfid向量化器实现文本数据提取")
print(tfidVector.fit_transform(text))
feature_names = tfidVector.get_feature_names_out()
feature_names2 = tfidVector2.get_feature_names_out()
print(feature_names)

# 存储每个文档中出现过的词汇的 TF-IDF 值
tfidf_values = []
for i in range(len(text)):
    doc_tfidf = {}
    doc_features = tfidf_matrix[i].nonzero()[1]  # 获取非零元素的索引
    for j in doc_features:
        word = feature_names[j]
        tfidf_value = tfidf_matrix[i, j]
        doc_tfidf[word] = tfidf_value
    tfidf_values.append(doc_tfidf)

tfidf_values2 = []
for i in range(len(text2)):
    doc_tfidf = {}
    doc_features = tfidf_matrix2[i].nonzero()[1]  # 获取非零元素的索引
    for j in doc_features:
        word = feature_names2[j]
        tfidf_value = tfidf_matrix2[i, j]
        doc_tfidf[word] = tfidf_value
    tfidf_values2.append(doc_tfidf)

tfidf_values.extend(tfidf_values2)
# 打印每个文档中词汇的 TF-IDF 值
for i, doc_tfidf in enumerate(tfidf_values):
    print(f"文档 {i+1}:")
    for word, tfidf in doc_tfidf.items():
        print(f"词 '{word}' 的 TF-IDF 值为: {tfidf:.4f}")

with open('/home/E22301339/depressionDetection/codes-author/emotions_ext/sub-emotions_2.0.json', 'r')as f:
    trust_cluster_vectos = json.load(f)

vectors = []
for item in tfidf_values:
    vector = []
    for word, tfidf in item.items():
        vector.append(trust_cluster_vectos[word])
    vectors.append(vector)

for i in range(len(tfidf_values)):
    tfidf_values[i] =list(tfidf_values[i].values())
print("================================================")   

tf_list=[]
for i in range(len(tfidf_values)):
    tf_vectors=[0]*300
    for j in range(len(tfidf_values[i])):
            vector=tfidf_values[i][j]*np.array(vectors[i][j])
            vector=vector.tolist()
            if(len(vector)==0):
                vector=[0]*300
            tf_vectors = [tf_vectors[i] + vector[i] for i in range(len(vector))]
    if(len(tfidf_values[i])!=0):
        tf_vectors=(np.array(tf_vectors)/len(tfidf_values[i])).tolist()
    tf_list.append(tf_vectors)
print("=============================================================")
with open('/home/E22301339/depressionDetection/processedData/average_features3.json', 'w')as f:
    json.dump(tf_list, f)
# %%