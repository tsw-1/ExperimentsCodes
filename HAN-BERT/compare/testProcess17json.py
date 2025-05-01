import os
import shutil
from natsort import natsorted

def merge_and_rename_files(train_folder, test_folder, target_folder):
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取 train 和 test 文件夹中的所有文件
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.json')]
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.json')]

    # 按自然顺序排列文件名
    train_files = natsorted(train_files)
    test_files = natsorted(test_files)

    # 根据文件名的标签进行分类
    train_files_label_0 = [f for f in train_files if f.endswith('_0.json')]
    train_files_label_1 = [f for f in train_files if f.endswith('_1.json')]
    test_files_label_0 = [f for f in test_files if f.endswith('_0.json')]
    test_files_label_1 = [f for f in test_files if f.endswith('_1.json')]


    # 重命名并复制文件到目标文件夹
    index = 0
    for file in train_files_label_0:
        source_path = os.path.join(train_folder, file)
        # 生成新的文件名
        new_filename = f"gpt_processed_{index:06d}_{'0' if '_0.json' in file else '1'}.json"
        target_path = os.path.join(target_folder, new_filename)
        
        shutil.copy(source_path, target_path)
        print(f"Copied and renamed {file} to {target_path}")
        index += 1

        
    for file in test_files_label_0:
        source_path = os.path.join(test_folder, file)
        # 生成新的文件名
        new_filename = f"gpt_processed_{index:06d}_{'0' if '_0.json' in file else '1'}.json"
        target_path = os.path.join(target_folder, new_filename)
        
        shutil.copy(source_path, target_path)
        print(f"Copied and renamed {file} to {target_path}")
        index += 1

    for file in train_files_label_1:
        source_path = os.path.join(train_folder, file)
        # 生成新的文件名
        new_filename = f"gpt_processed_{index:06d}_{'0' if '_0.json' in file else '1'}.json"
        target_path = os.path.join(target_folder, new_filename)
        
        shutil.copy(source_path, target_path)
        print(f"Copied and renamed {file} to {target_path}")
        index += 1

    for file in test_files_label_1:
        source_path = os.path.join(test_folder, file)
        # 生成新的文件名
        new_filename = f"gpt_processed_{index:06d}_{'0' if '_0.json' in file else '1'}.json"
        target_path = os.path.join(target_folder, new_filename)
        
        shutil.copy(source_path, target_path)
        print(f"Copied and renamed {file} to {target_path}")
        index += 1


# 使用示例
train_folder = "/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt/train_gpt_json/gpt/train"  # 替换为train文件夹路径
test_folder = "/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt/train_gpt_json/gpt/test"    # 替换为test文件夹路径
target_folder = "/E22301339/HAN-BERT/eRisk2018/processed/combined_maxsim32_gpt/train_gpt_json"  # 替换为目标文件夹路径

merge_and_rename_files(train_folder, test_folder, target_folder)


