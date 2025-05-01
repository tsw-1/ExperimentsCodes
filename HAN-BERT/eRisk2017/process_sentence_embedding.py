# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
import nltk
from nltk.corpus import stopwords
from natsort import natsorted

# %%
sbert = SentenceTransformer('/E22301339/paraphrase-MiniLM-L6-v2')


# %%
'''
定义了一个名为"get_input_data"的函数，它接受文件路径和文件名作为输入，并返回文件中的文本数据和帖子数量。
'''
def get_input_data(file, path):
    post_num = 0
    posts = []
    for i in range(10):
        dom = xml.dom.minidom.parse(path + "/" + file+str(i+1)+".xml")
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        text = collection.getElementsByTagName('TEXT')
        # stop_words = set(stopwords.words('english'))
        for i in range(len(title)):
            post = title[i].firstChild.data + ' ' + text[i].firstChild.data
            # post=re.sub(r'http\S+', '', post)   # 去除url
            # post=post.lower()                   # 小写
            # post = re.sub(r'\d+', '', post)     # 去除所有数字
            # post=re.sub(r'\b\d{2}-\d{2}-\d{4}\b', '', post) # 将日期格式替换为空字符串
            post = re.sub('\n', ' ', post)
            # words = nltk.word_tokenize(post)
            # words = [word for word in words if word not in stop_words]  # 去除停用词
            # post = ' '.join(words)
            if len(post) > 0:
                posts.append(post.strip())
                post_num = post_num + 1
        posts=posts[::-1]
    return posts, post_num


# %%
train_posts = []
train_tags = []
train_mappings = []
test_posts = []
test_tags = []
test_mappings = []  
for base_path in ["negative_examples_anonymous", "negative_examples_test", "positive_examples_anonymous", "positive_examples_test"]:
    base_path = "/E22301339/HAN-BERT/eRisk2017/dataset/"+base_path
    filenames = natsorted(os.listdir(base_path)) # 对base_path(一个子数据集)下的文件列表进行排序并返回

    # 处理10个xml连接起来
    for i in range(len(filenames)):
        filenames[i] = filenames[i][:filenames[i].rfind("_")+1]
    filenames=natsorted(list(set(filenames)))
 
    for fname in filenames: #   一个子数据集下的文件(包多个帖子)
        posts, post_num = get_input_data(fname, base_path) #fname是文件名，base_bath是该文件前缀路径
        if "anonymous" in base_path:
            # 训练集
            train_mappings.append(list(range(len(train_posts), len(train_posts)+post_num))) # 映射一个文件中所有帖子的位置
            train_posts.extend(posts) # extend()用于在列表尾部一次性追加另一个序列中的多个值
            train_tags.append(int("positive" in base_path)) # 记录训练集中阳性(帖子)文件的标签,值为"1"
        else:
            # 测试集
            test_mappings.append(list(range(len(test_posts), len(test_posts)+post_num)))
            test_posts.extend(posts)
            test_tags.append(int("positive" in base_path))


# %%
'''
使用"sbert"模型对训练集和测试集中的所有帖子进行编码，将编码的结果存储到变量"train_embs"和"test_embs"中。
'''
train_embs = sbert.encode(train_posts, convert_to_tensor=False)
train_embs.shape


test_embs = sbert.encode(test_posts, convert_to_tensor=False)
test_embs.shape


'''
将包含训练集和测试集的字典对象序列化为二进制对象，并存储到文件中。
'''
with open("/E22301339/HAN-BERT/eRisk2017/processed/miniLM_L6_embs.pkl", "wb") as f:
    pickle.dump({                                 # pickle.dump(): 将对象序列化成二进制对象；
        "train_posts": train_posts,               # 这里是将包含训练集和测试集的字典对象序列化为二进制对象，并存储到文件中。
        "train_mappings": train_mappings,
        "train_labels": train_tags,
        "train_embs": train_embs,
        "test_posts": test_posts,
        "test_mappings": test_mappings,
        "test_labels": test_tags,
        "test_embs": test_embs
    }, f)

"""
Do clustering
"""


# with open("processed/miniLM_L6_embs.pkl", "rb") as f:
#     data = pickle.load(f)

# train_posts = data["train_posts"]
# train_mappings = data["train_mappings"]
# train_tags = data["train_labels"]
# train_embs = data["train_embs"]
# test_posts = data["test_posts"]
# test_mappings = data["test_mappings"]
# test_tags = data["test_labels"]
# test_embs = data["test_embs"]

# from cluster_summary import get_kmeans_centroid_ids, get_cluster_summary

# user_posts1 = [train_posts[i] for i in train_mappings[0]]
# user_embs1 = train_embs[train_mappings[0]]
# summaries1 = get_cluster_summary(user_posts1, user_embs1, K=8)
# print("\n".join(summaries1))

# %% [markdown]
# run clustering and save the results

# for K in [8, 16, 32, 64]:
# for K in [16]:
#     os.makedirs(f"./processed/kmeans{K}", exist_ok=True)
#     os.makedirs(f"./processed/kmeans{K}/train", exist_ok=True)
#     os.makedirs(f"./processed/kmeans{K}/test", exist_ok=True)
#     for id0, members in enumerate(tqdm(train_mappings, desc=f"K={K}, train")):
#         user_posts1 = [train_posts[i] for i in members]
#         user_embs1 = train_embs[members]
#         label1 = train_tags[id0]
#         summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
#         with open(f"./processed/kmeans{K}/train/{id0:06}_{label1}.txt", "w") as f:
#             f.write("\n".join(summaries1))
#     for id0, members in enumerate(tqdm(test_mappings, desc=f"K={K}, test")):
#         user_posts1 = [test_posts[i] for i in members]
#         user_embs1 = test_embs[members]
#         label1 = test_tags[id0]
#         summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
#         with open(f"./processed/kmeans{K}/test/{id0:06}_{label1}.txt", "w") as f:
#             f.write("\n".join(summaries1))

#%%
# nltk.download('stopwords')
post='I am a gentleman , this is a good day!'
stop_words = set(stopwords.words('english'))
words = nltk.word_tokenize(post)
words = [word for word in words if word not in stop_words]  # 去除停用词
post = ' '.join(words)
print(post)

# %%
