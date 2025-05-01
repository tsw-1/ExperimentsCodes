# total_user_vectorpost={}
# vectors=[1,2,3,4,5]
# total_user_vectorpost['user']=vectors
# print(total_user_vectorpost)


# dict={}
# for i in range(5):
#     dict[i]=i
# print(dict)
# import json
# with open('/home/E22301339/depressionDetection/processedData/total_user_emotionpost5.json','r')as f:
#     data=json.load(f)
# print("================================")

# sample = [
#     "positive233",
#     "positive61",
#     "surprise55",
#     "trust121"
# ]

# merged_string = ' '.join(sample).strip()

# print(merged_string)

import numpy as np
import scipy.sparse as sp
import json

# # 创建稀疏矩阵
# data = np.array([1, 2, 3])
# row = np.array([0, 1, 2])
# col = np.array([0, 1, 2])
# matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3))





# # 假设 X_train 是一个 csr_matrix 对象
# X_train = sp.csr_matrix([[1, 2], [3, 4]])
# print(X_train)

# # 将 csr_matrix 转换为稠密矩阵 ndarray
# dense_matrix = X_train.toarray()
# print(type(dense_matrix))
# print(dense_matrix.tolist())
# # 存储稠密矩阵为 JSON 格式
# data = {}
# data['train'] = dense_matrix.tolist()

# # 将 data 对象转换为 JSON 字符串
# json_data = json.dumps(data)

# # 打印 JSON 字符串
# print(json_data)


with open('/home/E22301339/depressionDetection/processedData/total_user_wordpost2.json','r')as f:
    data=json.load(f)
print('==============')

