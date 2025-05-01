import os
import json

def txt_to_json(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
            
            # 读取txt文件内容并转换为json
            with open(input_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                data = {str(i+1): line.strip() for i, line in enumerate(lines)}
            
            # 写入json文件
            with open(output_file_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

# 示例用法
train_input_folder = "/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32/train"
train_output_folder = "/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32/train2"
test_input_folder = "/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32/test"
test_output_folder = "/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32/test2"

txt_to_json(train_input_folder, train_output_folder)
txt_to_json(test_input_folder, test_output_folder)
