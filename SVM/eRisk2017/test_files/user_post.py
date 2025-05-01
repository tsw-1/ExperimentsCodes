import json
import re
with open('/home/E22301339/depressionDetection/user_posts.json','r') as f:
    user_post=json.load(f)
first_user=user_post['train_subject64']
for item in first_user:
    for post in item:
        post=re.sub(r'http\S+', '', post)   # 去除url
        post=post.lower()                   # 小写
        post=re.sub(r'[^\w\s]', '', post)   #   匹配除了字母、数字和空白字符以外的所有字符（即标点符号），替换为空字符
        print(post)
        print('================================================================')





