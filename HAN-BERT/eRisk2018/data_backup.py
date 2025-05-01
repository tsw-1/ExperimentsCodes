from collections import defaultdict
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict
import test_random
from natsort import natsorted
import random
'''
这段代码主要是用PyTorch Lightning实现了两个数据集类和对应的数据模块类。
'''

class FlatDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir2):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            sample["text"] = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

class  HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64):
        # python中的断言语句，如果条件满足预期程序则会继续执行下面的语句，否则会抛出异常
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        # file_list = os.listdir(input_dir2)  # 获取文件列表
        # random.shuffle(file_list)           # 随机打乱列表顺序
            # tqdm可以在循环中显示进度条
        for fname in tqdm(os.listdir(input_dir2)):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            posts = open(os.path.join(input_dir2, fname), encoding="utf-8").read().strip().split("\n")[:max_posts]
            # tokenizer接受一个帖子列表posts，将其编码为一个字典，其中包括输入ids、注意力掩码、token类型ids等信息
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                    sample[k] = v
            self.data.append(sample)
            self.labels.append(label)
        print("---------------debug-------------------")

    def __len__(self) -> int:
        return len(self.data)
# 当使用 DataLoader 加载数据时，会自动调用 __getitem__() 方法来获取每一个 batch 对应的数据和标签
    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_hier(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            user_feats[k] = torch.LongTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    return processed_batch, labels

'''
定义了一个 PyTorch Lightning 的 DataModule，它被用来加载训练和测试数据集。
初始化时，它接受四个参数:   bs 示 batch size,input_dir 是数据集的路径,tokenizer 是一个用于将文本转化为token的tokenizer对象,max_len 表示序列的最大长度
'''
class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    # setup 方法被用于设置数据集,当策略为fit时，训练集和测试集被初始化
    def setup(self, stage):
        if stage == "fit":
            self.train_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.val_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        
        elif stage == "test":
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    # train_dataloader 和 val_dataloader 方法分别返回训练集和测试集的 DataLoader。
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)



def infer_preprocess(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch[k] = torch.LongTensor(batch[k])
    return batch


