import json
import fasttext
from sentence_transformers import SentenceTransformer, util
import numpy as np

print("Loading................................................................")
# 加载预训练好的 FastText 模型
model = fasttext.load_model('/home/E22301339/fastText/fastText/cc.en.300.bin')
total_user_vectorpost={}

with open ('/home/E22301339/depressionDetection/codes-author/data/total_user_wordpost_Trainpart_4.json','r')as f:
    user_dict=json.load(f)

with open('/home/E22301339/depressionDetection/codes-author/emotions_ext/sub-emotions.json','r')as f:
    total_clusters_vector = json.load(f)
    clusters_vectors=list(total_clusters_vector.values())
    keys=list(total_clusters_vector.keys())

count_user=0
for user in user_dict:
    emotions=[]
    i=0
    count_user+=1
    length=len(user_dict[user])
    for word in user_dict[user]:
        i+=1
        word_vector=model[word] 
        cosine_scores = util.pytorch_cos_sim(clusters_vectors, word_vector)
        cluster_score_list=cosine_scores.flatten().tolist()
        cluster_score_list=np.array(cluster_score_list)

        max_idx = np.argmax(cluster_score_list)
        max_cluster_score=cluster_score_list[max_idx]
        # print(str(count_user)+str(max_cluster_score)+'\t'+str(i)+'/'+str(length))
        print("{}\t{} \t {}/{}".format(count_user, max_cluster_score, i, length))

        index=max_idx
        key=keys[index]
        emotions.append(key)
    total_user_vectorpost[user]=emotions

with open('/home/E22301339/depressionDetection/codes-author/data/total_user_emotionpost_train4.json','w')as f:
    json.dump(total_user_vectorpost, f)
    
print("end................................................................") 
