# # with open('./dataset/negative_examples_anonymous','r+') as f:
# import os
# import pickle
# import torch
# # path="/home/E22301339/scale_early_depress_detect-main/eRisk2017/dataset"
# # files=os.listdir(path)
# # print(files)
# # data={"name":['x','y','z'],'age':[18,20,22]}
# # with open("./test/miniLM_L6_embs.pkl", "wb") as f:
# #     pickle.dump(data, f)                              # pickle.dump(): 将对象序列化成二进制对象；

# print(torch.cuda.device_count())
# list=os.listdir("/home/E22301339/scale_early_depress_detect-main/eRisk2017/dataset/positive_examples_test")
# print("========")

import re
import os
# filename = "train_subject64_10.xml"
# i = "3"
# new_filename = re.sub(r"_\d+(?=\.)", "_" + i, filename)
# print(new_filename)

# filename = "train_subject64_1.xml"
# new_filename = filename[:filename.rfind("_")+1]
# print(new_filename)

# i=5
# filename = "train_subject64_"
# new_filename = filename+str(i)+'.xml'
# print(new_filename)

list=sorted(os.listdir('/home/E22301339/scale_early_depress_detect-main/eRisk2017/dataset/positive_examples_test'))
print("=======")