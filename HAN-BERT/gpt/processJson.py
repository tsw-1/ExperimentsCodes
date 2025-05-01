import os
import json
from natsort import natsorted

# 定义处理单个 JSON 文件的函数
def process_single_json_file(filepath, output_filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        result = []
        for post_id, post_content in data.items():
            result.append({post_id: post_content})
    
    # 将处理后的结果写入一个新的 JSON 文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# 处理目录中的所有 JSON 文件
def process_json_files_in_directory(directory):
    for subdir in ['test', 'train']:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for file in natsorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, file)
            if filepath.endswith(".json"):
                output_filepath = os.path.join(subdir_path, f"processed_{file}")
                process_single_json_file(filepath, output_filepath)

process_json_files_in_directory('/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32_json')
