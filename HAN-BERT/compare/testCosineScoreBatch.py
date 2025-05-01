import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from natsort import natsorted

# 初始化模型
sbert = SentenceTransformer('/E22301339/paraphrase-MiniLM-L6-v2')
# 定义抑郁量表模板
questionaire_single = [ 
    "I feel sad.",
    "I am discouraged about my future.",
    "I always fail.",
    "I don't get pleasure from things.",
    "I feel quite guilty.",
    "I expected to be punished.",
    "I am disappointed in myself.",
    "I always criticize myself for my faults.",
    "I have thoughts of killing myself.",
    "I always cry.",
    "I am hard to stay still.",
    "It's hard to get interested in things.",
    "I have trouble making decisions.",
    "I feel worthless.",
    "I don't have energy to do things.",
    "I have changes in my sleeping pattern.",
    "I am always irritable.",
    "I have changes in my appetite.",
    "I feel hard to concentrate on things.",
    "I am too tired to do things.",
    "I have lost my interest in sex."
]
depression_texts = [     #抑郁症模板中的第一组：3个显性抑郁的表达组成，对应患者的抑郁情况。
    "I feel depressed.",
    "I am diagnosed with depression.",
    "I am treating my depression."
]
questionaire_single_embs = sbert.encode(questionaire_single)
depression_embs = sbert.encode(depression_texts)
# 定义要处理的文件夹
folders = ['test']

# 遍历文件夹
for folder in folders:
    folder_path = os.path.join("/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt", folder)
    
    # 遍历文件夹中的每个JSON文件
    for filename in natsorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取JSON文件
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 遍历每个帖子
            for post in data:
                # 获取帖子的文本内容（第一个key的value）
                post_id = list(post.keys())[0]
                post_text = post[post_id]
                
                # 计算帖子的嵌入向量
                post_embedding = np.array([sbert.encode(post_text)])
                
                # 计算与抑郁症相关的文本的相似度
                depression_pair_sim = cosine_similarity(post_embedding, depression_embs)
                dimension_sim_single = cosine_similarity(post_embedding, questionaire_single_embs)
                combined_sim = np.concatenate([depression_pair_sim, dimension_sim_single], axis=1)
                
                # 找出最大值
                max_value = float(np.max(combined_sim))
                
                # 添加scale_score到post中
                post['scale_score'] = max_value
            
            # 将更新后的数据写回JSON文件
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

print("Done processing all files.")
