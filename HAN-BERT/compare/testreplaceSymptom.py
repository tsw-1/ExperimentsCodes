import os

def replace_line_in_file(file_path, old_line, new_line):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        for line in lines:
            # 如果行与 old_line 匹配，则替换为 new_line
            if line.strip() == old_line:
                file.write(new_line + '\n')
            else:
                file.write(line)

def process_folder(folder_path, old_line, new_line):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            replace_line_in_file(file_path, old_line, new_line)

# 定义文件夹路径
train_folder = '/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt/train_Symptoms'
test_folder = '/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt/test_Symptoms'

# 需要替换的行
old_text = "I have no symptoms of depression."
new_text = "No relevant descriptions."

# 处理两个文件夹
process_folder(train_folder, old_text, new_text)
process_folder(test_folder, old_text, new_text)

print("替换完成")
