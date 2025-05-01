import os
import shutil
from natsort import natsorted
# 定义文件夹路径
train_folder = '/E22301339/HAN-BERT/eRisk2017/processed/top16_twoDimenScore/111/trainSymptoms'
test_folder = '/E22301339/HAN-BERT/eRisk2017/processed/top16_twoDimenScore/111/testSymptoms'

# 获取train文件夹中最后一个文件的编号
train_files = natsorted(os.listdir(train_folder))
if train_files:
    last_train_file = train_files[-1]
    base_num = int(last_train_file.split('_')[0]) + 1
else:
    base_num = 0

# 获取test文件夹中的所有文件
test_files = natsorted(os.listdir(test_folder))

# 处理并移动test文件
for i, test_file in enumerate(test_files):
    new_filename = f"{base_num + i:06d}_{test_file.split('_')[1]}_Symptoms.txt"
    shutil.move(os.path.join(test_folder, test_file), os.path.join(train_folder, new_filename))

print("文件已成功合并。")
