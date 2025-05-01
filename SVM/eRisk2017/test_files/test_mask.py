import json
import fasttext
from sentence_transformers import SentenceTransformer, util
import numpy as np

str="Success grows out of struggles to overcome difficulties"
str2="I am too tired to do things"
str=str.split()
str2=str2.split()
mask_list=[]
mask_list2=[]

model = fasttext.load_model('/home/E22301339/fastText/fastText/cc.en.300.bin')

with open('/home/E22301339/depressionDetection/codes-author/emotions_ext/sub-emotions.json','r')as f:
    total_clusters_vector = json.load(f)
    clusters_vectors=list(total_clusters_vector.values())
    keys=list(total_clusters_vector.keys())

for word in str:
    word_vector=model[word] 
    cosine_scores = util.pytorch_cos_sim(clusters_vectors, word_vector)
    cluster_score_list=cosine_scores.flatten().tolist()
    cluster_score_list=np.array(cluster_score_list)

    max_idx = np.argmax(cluster_score_list)
    max_cluster_score=cluster_score_list[max_idx]

    index=max_idx
    key=keys[index]
    mask_list.append(key)
    print("=====")

for word in str2:
    word_vector=model[word] 
    cosine_scores = util.pytorch_cos_sim(clusters_vectors, word_vector)
    cluster_score_list=cosine_scores.flatten().tolist()
    cluster_score_list=np.array(cluster_score_list)

    max_idx = np.argmax(cluster_score_list)
    max_cluster_score=cluster_score_list[max_idx]

    index=max_idx
    key=keys[index]
    mask_list2.append(key)
    print("=====")
print(mask_list)
print(mask_list2)