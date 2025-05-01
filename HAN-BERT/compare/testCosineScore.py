import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
questionaire_single = [ #   BDI-II 抑郁症测量表定义的症状----抑郁症模板中的第二组
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
sbert = SentenceTransformer('/E22301339/paraphrase-MiniLM-L6-v2')   #sentence-transformers模型

questionaire_single_embs = sbert.encode(questionaire_single)  #计算表示
depression_embs = sbert.encode(depression_texts)
test_post="No relevant descriptions."
post_embedding=sbert.encode(test_post).reshape(1, -1)
depression_pair_sim = cosine_similarity(post_embedding, depression_embs)  
dimension_sim_single = cosine_similarity(post_embedding, questionaire_single_embs)
combined_sim = np.concatenate([depression_pair_sim, dimension_sim_single], axis=1)  
max_score =np.max(combined_sim)
print(max_score)