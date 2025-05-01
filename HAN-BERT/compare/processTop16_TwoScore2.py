import os
import json
from natsort import natsorted

# 定义要处理的文件夹
folders = ['train']

# 遍历文件夹
for folder in folders:
    folder_path = os.path.join("/E22301339/HAN-BERT/eRisk2017/processed/gpt", folder)
    
    # 遍历文件夹中的每个JSON文件
    for filename in natsorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取排序好的JSON文件
            with open(file_path, 'r') as f:
                sorted_data = json.load(f)
            
            # 筛选出前16的帖子
            top_16_posts = sorted_data[:16]
            
            # 准备要写入txt文件的内容
            txt_lines_posts = []
            txt_lines_symptoms = []
            
            for post in top_16_posts:
                # 提取帖子的文本内容
                post_content = list(post.values())[0]
                txt_lines_posts.append(post_content)
                
                # 提取Symptoms部分
                symptoms = post.get("Symptoms", [])
                if not symptoms:
                    # 如果Symptoms列表为空
                    txt_lines_symptoms.append("I have no symptoms of depression.")
                else:
                   # 将所有症状合并成一行
                    symptoms_line = ", ".join(f"I feel {symptom.lower()}" for symptom in symptoms)
                    txt_lines_symptoms.append(symptoms_line)

            # # 保存前16条帖子的内容到txt文件，每条内容占一行
            # txt_filename_posts = filename.replace('gpt_processed_', '').replace('.json', '.txt')
            # txt_path_posts = os.path.join(folder_path, txt_filename_posts)
            # with open(txt_path_posts, 'w') as txt_file:
            #     for line in txt_lines_posts:
            #         txt_file.write(line + "\n")

            # 保存前16条帖子的Symptoms内容到txt文件，每条内容占一行
            txt_filename_symptoms = filename.replace('gpt_processed_', '').replace('.json', '_Symptoms.txt')
            txt_path_symptoms = os.path.join(folder_path, txt_filename_symptoms)
            with open(txt_path_symptoms, 'w') as txt_file:
                for line in txt_lines_symptoms:
                    txt_file.write(line + "\n")
