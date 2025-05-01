import os
from natsort import natsorted

# 文件夹路径
input_dir = "/E22301339/HAN-BERT/eRisk2017/processed/top16_twoDimenScore/111/trainSymptoms"  # 替换为你的train文件夹路径

# 读取文件名并排序（自然排序）
file_names = natsorted(os.listdir(input_dir))

# 按照标签分组
zero_labels = [fname for fname in file_names if fname.endswith('_0_Symptoms.txt')]
one_labels = [fname for fname in file_names if fname.endswith('_1_Symptoms.txt')]

# 合并列表，保证0标签的文件在前面
sorted_files = zero_labels + one_labels

# 重命名文件，使编号连续
for i, fname in enumerate(sorted_files):
    new_name = f"{i:06}_{fname.split('_')[1]}_Symptoms.txt"  # 保持文件名格式一致并加上 .txt 后缀
    os.rename(os.path.join(input_dir, fname), os.path.join(input_dir, new_name))
