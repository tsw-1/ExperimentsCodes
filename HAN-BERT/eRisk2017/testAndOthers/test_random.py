import os
from natsort import natsorted
import random

# 对训练集执行欠采样
def under_random():
    train=os.listdir('/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim16/train')
    train_positive=natsorted(train)[-83:]
    train_negative=natsorted(train)[:403]
    under_samples=random.sample(train_negative,83)
    under_samples=under_samples+train_positive
    return under_samples
