from sentence_transformers import SentenceTransformer
from transformers import file_utils

# 加载模型
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 获取模型缓存路径
model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
cache_dir = file_utils.default_cache_path
print(f'Model is cached in: {cache_dir}/{model_name}')
