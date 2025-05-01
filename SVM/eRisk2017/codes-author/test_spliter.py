#%%
import json
with open('/home/E22301339/depressionDetection/processedData/total_user_wordpost.json', 'r') as f:
    posts = json.load(f)

# 将dict的key按照顺序保存到一个列表中
keys_list = list(posts.keys())

# 将前9份的大小设置为50，最后一份的大小设置为36
part_sizes = [50] * 9 + [36]

# 分割样本数据
parts = []
start_index = 0
for part_size in part_sizes:
    part = {k: posts[k] for k in keys_list[start_index: start_index + part_size]}
    parts.append(part)
    start_index += part_size

# 保存分割后的样本数据
for i, part in enumerate(parts, start=1):
    file_path = f'/home/E22301339/depressionDetection/codes-author/data/total_user_wordpost_trainpart_{i}.json'
    with open(file_path, 'w') as f:
        json.dump(part, f)
        print(f"Part {i} saved to {file_path}")

print("================================================")
#%%
import json

with open('/home/E22301339/depressionDetection/processedData/total_user_wordpost2.json', 'r') as f:
    posts = json.load(f)

# 将dict的key按照顺序保存到一个列表中
keys_list = list(posts.keys())

# 将前8份的大小设置为50，最后一份的大小设置为2
part_sizes = [50] * 8 + [1]

# 分割样本数据
parts = []
start_index = 0
for part_size in part_sizes:
    part = {k: posts[k] for k in keys_list[start_index: start_index + part_size]}
    parts.append(part)
    start_index += part_size

# 保存分割后的样本数据
for i, part in enumerate(parts, start=1):
    file_path = f'/home/E22301339/depressionDetection/codes-author/data/total_user_wordpost_trainpart_{i}.json'
    with open(file_path, 'w') as f:
        json.dump(part, f)
        print(f"Part {i} saved to {file_path}")

print("================================================")

# %%
import json

with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train1.json','r')as f:
    train=json.load(f)
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train2.json','r')as f:
    train2=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train3.json','r')as f:
    train3=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train4.json','r')as f:
    train4=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train5.json','r')as f:
    train5=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train6.json','r')as f:
    train6=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train7.json','r')as f:
    train7=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train8.json','r')as f:
    train8=json.load(f)   
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train9.json','r')as f:
    train9=json.load(f)     
with open('/home/E22301339/depressionDetection/codes-author/data/data_masked2/total_user_emotionpost_train10.json','r')as f:
    train10=json.load(f)    
train.update(train2)
train.update(train3)
train.update(train4)
train.update(train5)
train.update(train6)
train.update(train7)
train.update(train8)
train.update(train9)
train.update(train10)
with open("/home/E22301339/depressionDetection/codes-author/data/train2.json",'w')as f:
    json.dump(train,f)

# %%
