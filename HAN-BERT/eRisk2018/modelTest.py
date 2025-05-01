import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LightningInterface(pl.LightningModule):
    # 初始化方法，初始化了一个最佳的F1值和一个二分类阈值，同时定义了一个交叉熵损失函数
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        # 创建了一个二元交叉熵损失函数,用于在训练期间计算模型预测与目标标签之间的误差。
        # 这种损失函数通常用于二分类问题，其中每个样本都可以属于两个类别中的一个。
        self.criterion = nn.BCEWithLogitsLoss()
        

    # 训练步骤，接收一个批次的数据，计算模型的预测结果和损失函数值，并返回损失函数值和日志信息
    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y = batch
        y_hat = self(x) # forward的返回结果
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    # 验证步骤，接收一个批次的数据，计算模型的预测结果和损失函数值，并返回损失函数值、标签和概率值
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': self.criterion(y_hat, y), "labels": yy, "probs": yy_hat}
    # 验证步骤结束后的处理，计算平均损失和评价指标（包括准确率、精确率、召回率和F1分数），并更新最佳F1值，返回评价指标和日志信息。
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(float)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        self.best_f1 = max(self.best_f1, f1)
        if self.current_epoch == 0:  # prevent the initial check modifying it
            self.best_f1 = 0
        # return {'val_loss': avg_loss, 'val_acc': avg_acc, 'hp_metric': self.best_acc}
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_p': p, 'val_r': r, 'val_f1': f1, 'hp_metric': self.best_f1}
        # import pdb; pdb.set_trace()
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # 测试步骤，接收一个批次的数据，计算模型的预测结果和损失函数值，并返回损失函数值、标签和概率值
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        print("test_step:",str(self.criterion(y_hat, y)))
        return {'test_loss': self.criterion(y_hat, y), "labels": yy, "probs": yy_hat}

    # 测试步骤结束后的处理，计算平均损失和评价指标（包括准确率、精确率、召回率和F1分数），返回评价指标
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(float)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print("预测概率值: ",all_probs.tolist())
        print("原数据标签: ",all_labels.tolist())
        print("测试集F1：{}\nPrecision：{}\nAccuracy：{}\navg_loss：{}\nRecall：{}".format(f1,p,acc,avg_loss,r))
        return {'test_loss': avg_loss, 'test_acc': acc, 'test_p': p, 'test_r': r, 'test_f1': f1}

    # 反向传播后的处理，可以在这里检查梯度。
    def on_after_backward(self):
        pass
    # 静态方法，用于添加模型特定的参数。
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser



class BERTHierClassifierTransAbs(nn.Module):
    # 模型初始化方法：初始化模型的结构、参数、组件等；只会在创建模型实例时执行一次
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type    # 指定预训练模型的类型，如BERT、RoBERTa等
        self.num_heads = num_heads      # Transformer编码器中多头注意力机制的头数
        self.num_trans_layers = num_trans_layers # Transformer编码器中的层数。
        self.pool_type = pool_type               # 池化类型，如何从每个帖子的特征中提取信息（使用第一个token的特征或平均池化）。
        self.post_encoder = AutoModel.from_pretrained(model_type)   # 帖子编码器：用于对每个帖子进行编码。
        if freeze:                                                  # 可选的冻结操作，如果freeze=True，则在训练中模型的参数不会更新。
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size      # 获取预训练模型的隐藏层维度，即每个帖子的特征表示的大小。
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))   # 初始化一个位置编码矩阵，用于为每个帖子的位置提供位置信息。
        nn.init.xavier_uniform_(self.pos_emb)                                   # 对位置编码矩阵进行初始化，使其值在合理范围内分布。
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu') # 定义一个Transformer编码器层，指定了输入维度、前馈网络的维度、多头注意力的头数和激活函数。
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers) # 将上述编码器层堆叠为多个层，形成最终的用户编码器。这个编码器处理多个帖子的特征，并将它们聚合为用户级别的特征。
        self.attn_ff = nn.Linear(self.hidden_dim, 1)                                          # 一个线性层，用于计算每个帖子的注意力分数，这些分数用于加权平均不同帖子的特征。
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)               # 在特征融合后应用的 dropout 操作，防止过拟合。
        self.clf = nn.Linear(self.hidden_dim, 1)                                              # 最终的分类器，用于将用户级别的特征映射到一个输出值（即二分类的logit）。
    
    def forward(self, batch, **kwargs):
        '''
        forward:模型的前向传播方法：定义了输入数据如何通过网络中的各层进行计算，并最终输出预测结果。每batch将数据输入模型时，forward 方法都会被调用。
        feats是一个列表,其中包含每个用户的特征向量,每个特征向量是一个加权平均后的帖子向量,权重由self.attn_ff计算。
        feats的形状为(batch_size, hidden_size),其中batch_size是一批用户的数量,hidden_size是特征向量的维度。
        '''
        feats = []                  # 存储每个用户的特征表示
        attn_scores = []            # 存储每个用户的注意力分数。
        for user_feats in batch:    # batch 是一个包含多个用户数据的列表，每个 user_feats 包含一个用户的所有帖子信息
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])   # 对用户帖子进行编码，利用三个key的val
            if self.pool_type == "first":         # 池化帖子特征，两种策略 
                x = post_outputs.last_hidden_state[:, 0:1, :] # 1. first取每个帖子的第一个 token 的输出作为帖子特征
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1) # 2.mean:使用mean_pooling对整个帖子进行均值池化，以得到一个帖子级别的表示。
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1) # 将位置编码添加到每个帖子特征上
            x = self.user_encoder(x).squeeze(1)               # 使用用户编码器对帖子特征进行编码
            attn_score = torch.softmax(self.attn_ff(x).squeeze(-1), -1)     # 每个帖子的注意力分数
            feat = attn_score @ x                                           # 根据注意力分数对帖子特征进行加权平均
            feats.append(feat)                                              # 存储每个用户的特征表示
            attn_scores.append(attn_score)                                  # 存储每个用户的注意力分数
        feats = torch.stack(feats)                                          # 张量堆叠，方便后续批处理，dropout和clf分类
        x = self.dropout(feats)                                             # 对特征应用Dropout,防止过拟合
        logits = self.clf(x).squeeze(-1)                                    # 通过全连接层进行分类,输出预测的logits（未经过sigmoid函数）
        print("=================Debug=======================")
        return logits, attn_scores


'''
这段代码是一个PyTorch Lightning模型的定义, 是分类器的一个层次结构版本。它根据传递的参数选择使用哪个模型，然后，它通过将输入传递到模型来进行前向传播。在前向传播过程中，模型将其输出作为分类器的输出返回。
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
trans_abs" 是一种用户自定义的编码器选项，在这个代码中用来控制模型使用何种编码器。在这个模型中，支持以下四种选项：
    "none": 不使用用户自定义编码器，使用默认的 BERT 编码器。
    "trans": 使用自定义的 Transformer 编码器。
    "trans_abs": 在 "trans" 的基础上，对 Transformer 的输出进行绝对值处理。
    "trans_abs_avg": 在 "trans_abs" 的基础上，对 Transformer 的输出进行均值池化。
这些选项用于定义模型的不同结构，以提高模型性能。
'''
class HierClassifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", user_encoder="none", num_heads=8, num_trans_layers=2, freeze_word_level=False, pool_type="first", vocab_size=30522, **kwargs):
        super().__init__(threshold=threshold, **kwargs)

        self.model_type = model_type
        #   trans_abs表示使用一种基于 Transformer 的编码器，并且该编码器在最后一层使用绝对位置编码。
        if user_encoder == "trans_abs":
             # 创建了一个名为model的模型实例,类型为BERTHierClassifierTransAbs
            self.model = BERTHierClassifierTransAbs(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        self.lr = lr 
        self.save_hyperparameters() # 保存模型超参数到hparams.yaml文件下
        print(self.hparams) # 打印超参数信息

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-5)
        # parser.add_argument("--trans", action="store_true")
        parser.add_argument("--user_encoder", type=str, default="trans_abs")
        parser.add_argument("--pool_type", type=str, default="first")
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--num_trans_layers", type=int, default=4)
        parser.add_argument("--freeze_word_level", action="store_true")
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer