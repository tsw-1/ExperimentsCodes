import os
import json
from natsort import natsorted

# 读取单个 JSON 文件并返回帖子列表
def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 从每个 JSON 文件中筛选出风险最高的前16条帖子
def get_top_risk_posts(directory):
    for subdir in ['train', 'test']:  # 修改子文件夹列表为 train 和 test
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for file in natsorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, file)
            if filepath.endswith(".json"):
                top_risk_posts = []
                data = read_json_file(filepath)
                for post in data:
                    # 获取风险评分
                    content=next(iter(post.values()))
                    score = post.get("Scores", 0)
                    if score =="": 
                        score = 0
                    # 添加帖子和评分到列表中
                    top_risk_posts.append((content, score))
            # 按照评分降序排序，获取前16条风险最高的帖子
            top_risk_posts = sorted(top_risk_posts, key=lambda x: x[1], reverse=True)[:16]
            # 写入新的文本文件
            output_filename = file.replace("gpt_processed_", "").replace(".json", ".txt")
            output_filepath = os.path.join(subdir_path, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for post in top_risk_posts:
                    f.write(post[0] + "\n")
            print(f"Processed {file} and saved top 16 risk posts to {output_filename}")

# 示例调用
directory_path = '/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32_json/gpt'
top_risk_posts = get_top_risk_posts(directory_path)

