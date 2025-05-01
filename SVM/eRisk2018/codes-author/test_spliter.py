#%%
import json
with open('/home/E22301339/depressionDetection-eRisk2018/processedData/total_user_wordpost2.json', 'r') as f:
    posts = json.load(f)

# 将dict的key按照顺序保存到一个列表中
keys_list = list(posts.keys())

# 将前8份的大小设置为100，最后一份的大小设置为20
part_sizes = [100] * 8 + [20]

# 分割样本数据
parts = []
start_index = 0
for part_size in part_sizes:
    part = {k: posts[k] for k in keys_list[start_index: start_index + part_size]}
    parts.append(part)
    start_index += part_size

# 保存分割后的样本数据
for i, part in enumerate(parts, start=1):
    file_path = f'/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_wordpost_testpart_{i}.json'
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
import json

with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test1.json','r')as f:
    test=json.load(f)
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test2.json','r')as f:
    test2=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test3.json','r')as f:
    test3=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test4.json','r')as f:
    test4=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test5.json','r')as f:
    test5=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test6.json','r')as f:
    test6=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test7.json','r')as f:
    test7=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test8.json','r')as f:
    test8=json.load(f)   
with open('/home/E22301339/depressionDetection-eRisk2018/codes-author/data/total_user_emotionpost_test9.json','r')as f:
    test9=json.load(f)       
test.update(test2)
test.update(test3)
test.update(test4)
test.update(test5)
test.update(test6)
test.update(test7)
test.update(test8)
test.update(test9)
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/test.json",'w')as f:
    json.dump(test,f)
# %%
import json
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/train.json",'r')as f:
    train=json.load(f)
with open("/home/E22301339/depressionDetection/codes-author/data/test2.json",'r')as f:
    train2=json.load(f)
train.update(train2)    
with open("/home/E22301339/depressionDetection-eRisk2018/codes-author/data/train.json",'w')as f:
    json.dump(train,f)
# %%
