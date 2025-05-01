
很慢很慢，又占内存
#%%
from gensim.models import fasttext
model = fasttext.load_facebook_vectors("/home/E22301339/fastText/fastText/cc.en.300.bin") 


快很多，不占内存
#%%
import fasttext
model = fasttext.load_model('/home/E22301339/fastText/fastText/cc.en.300.bin')
#%%
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

constant = 2

# 使用循环对矩阵每个元素进行乘法运算
result = []
for row in matrix:
    new_row = [constant * element for element in row]
    result.append(new_row)
print(result)
# %%
import numpy as np

matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

constant = 2

# 将列表转换为NumPy数组
matrix_np = np.array(matrix)

# 常数乘以矩阵
result_np = constant * matrix_np

print("结果矩阵:")
print(result_np)

# %%

import numpy as np

list1 = [[1, 2, 3], [4, 5, 6]]
list2 = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]

result = []
for i in range(len(list1)):
    temp = []
    for j in range(len(list1[i])):
        temp.append(np.average(np.array(list1[i]) * np.array(list2[i][j])))
    result.append(temp)

print(result)


# %%
import numpy as np

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3=[]

result = np.add(list1, list2)
print(result)
print(np.add(list1, list3))
# %%
import json
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/train2.json",'r')as f:
    train=json.load(f)
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/test2.json",'r')as f:
    test=json.load(f) 
train.update(test)
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/train2.json",'w')as f:
    json.dump(train,f)
print("========")    