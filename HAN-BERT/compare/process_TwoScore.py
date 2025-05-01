import os
import json
from natsort import natsorted

# 定义要处理的文件夹
folders = [ 'train_gpt_json']

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

           # 预处理Scores，将非数值类型和空字符串转换为0
            for post in data:
                if isinstance(post['Scores'], str) or post['Scores'] == "":
                    try:
                        post['Scores'] = int(post['Scores'])
                    except ValueError:
                        post['Scores'] = 0     

            # 对data中的每个帖子按scale_score降序、Scores降序排序
            sorted_data = sorted(data, key=lambda x: (x['scale_score'], x['Scores']), reverse=True)
            # 将排序后的数据覆盖写入原JSON文件
            with open(file_path, 'w') as f:
                json.dump(sorted_data, f, indent=4)

print("Done processing all files.")
