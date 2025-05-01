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
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        # import pdb; pdb.set_trace()
        # self.log('lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0], on_step=True)
        # self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # def training_epoch_end(self, output) -> None:
    #     self.log('lr', self.hparams.lr)

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
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
    # 静态方法，用于添加模型特定的参数。
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class Classifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", **kwargs):
        super().__init__(threshold=threshold, **kwargs)

        self.model_type = model_type
        self.model = BERTFlatClassifier(model_type)
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(**x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        self.model_type = model_type
        # binary classification
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, 1)
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # import pdb; pdb.set_trace()
        x = outputs.last_hidden_state[:, 0, :]
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x).squeeze(-1)
        return logits



class BERTHierClassifierSimple(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        self.model_type = model_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        self.attn_ff = nn.Linear(self.post_encoder.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.post_encoder.config.hidden_size, 1)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, hidden_size]
            x = post_outputs.last_hidden_state[:, 0, :]
            # [num_posts, ]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(-1), -1)
            # weighted sum [hidden_size, ]
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)    
        # [bs, num_posts]
        return logits, attn_scores

class BERTHierClassifierTrans(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        # batch_first = False
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            # [num_posts, ]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(-1), -1)
            # weighted sum [hidden_size, ]
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)
        # [bs, num_posts]
        return logits, attn_scores

class BERTHierClassifierTransAbs(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        #   帖子编码器，它使用预训练的BERT模型进行帖子的编码
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        #   位置编码矩阵，用于为每个帖子的位置提供编码信息，以便Transformer编码器能够处理序列中不同位置的信息
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        #   用户编码器，它使用多层的Transformer Encoder进行用户的编码
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        #   是一个线性层，用于计算用户编码器输出中每个帖子的权重，以便将所有帖子的信息合并为单个用户向量
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        #   分类器，它将用户编码的输出映射到一个单一的输出值，用于分类任务
        self.clf = nn.Linear(self.hidden_dim, 1)    # self.clf 是一个 nn.Linear 的实例
    
    def forward(self, batch, **kwargs):
        '''
        feats是一个列表，其中包含每个用户的特征向量。
        每个特征向量是一个加权平均后的帖子向量，权重由self.attn_ff计算。
        feats的形状为(batch_size, hidden_size)，其中batch_size是一批用户的数量，hidden_size是特征向量的维度。
        '''
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            # [num_posts, ]
            # attn_score是一个包含每个帖子在用户级别上的重要性分数的张量，形状为[num_posts, ],用作权重矩阵
            attn_score = torch.softmax(self.attn_ff(x).squeeze(-1), -1) # torch.Size([16])
            # weighted sum [hidden_size, ]
            # @表示矩阵乘法  x 是一个形状为 [num_posts, hidden_size] 的张量 torch.Size([16, 768]) 每一行对应一个帖子的表示
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)
        # [bs, num_posts] 
        # print('logits的值： {}\n===========================================\nattn_scores的值：{}'.format(logits,attn_score))
        # print('\033[32mlogits的值： {}\n===========================================\nattn_scores的值：{}\033[0m'.format(logits,attn_score))

        print("========================================")
        return logits, attn_scores

class BERTHierClassifierTransAbsAvg(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, batch, **kwargs):
        feats = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            feat = x.mean(0)
            feats.append(feat)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)
        # [bs, num_posts]
        return logits

class GRUAttnModel(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_size, gru_layers=1, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.gru_layers = gru_layers
        self.word_rnn = nn.GRU(emb_size, hidden_size, gru_layers, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(2*hidden_size, attn_size)
        self.word_context_vector = nn.Linear(attn_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embedded, attention_mask=None):
        if attention_mask is not None:
            words_per_sentence = attention_mask.sum(1).tolist()
        else:
            words_per_sentence = [embedded.shape[1]] * embedded.shape[0]
        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(embedded,
                                            lengths=words_per_sentence,
                                            batch_first=True,
                                            enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words)  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)
        return sentences

class GRUHANClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=100, hidden_size=100, attn_size=100,
                 gru_layers=1, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.gru_layers = gru_layers

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # self.emb = nn.Embedding(vocab_size, emb_size)
        self.post_encoder = GRUAttnModel(emb_size, hidden_size, attn_size, gru_layers, dropout)
        self.user_encoder = GRUAttnModel(2*hidden_size, hidden_size, attn_size, gru_layers, dropout)
        self.clf = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch, **kwargs):
        feats = []
        for user_feats in batch:
            embedded = self.emb(user_feats["input_ids"])
            # [num_posts, seq_len, emb_size] -> [num_posts, 2*hidden_size]:
            x = self.post_encoder(embedded, user_feats["attention_mask"]).unsqueeze(0)
            post_attention_mask = (user_feats['attention_mask'].sum(1) > 2).float().unsqueeze(0)
            feat = self.user_encoder(x, post_attention_mask).view(-1) # [2*hidden_size, ]
            feats.append(feat)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)
        return logits


'''
这段代码是一个PyTorch Lightning模型的定义。
该模型是分类器的一个层次结构版本。它根据传递的参数选择使用哪个模型。
然后，它通过将输入传递到模型来进行前向传播。在前向传播过程中，模型将其输出作为分类器的输出返回。
'''
'''
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
        if user_encoder == "trans":
            self.model = BERTHierClassifierTrans(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        #   trans_abs 表示使用一种基于 Transformer 的编码器，并且该编码器在最后一层使用绝对位置编码。
        elif user_encoder == "trans_abs":
             # 创建了一个名为model的模型实例,类型为BERTHierClassifierTransAbs
            self.model = BERTHierClassifierTransAbs(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "trans_abs_avg":
            self.model = BERTHierClassifierTransAbsAvg(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == 'han_gru':
            self.model = GRUHANClassifier(vocab_size)
        else:
            self.model = BERTHierClassifierSimple(model_type)
        self.lr = lr
        # self.lr_sched = lr_sched
        # 保存模型超参数到hparams.yaml文件下
        self.save_hyperparameters()
        # 打印超参数信息
        print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        # print("\033[91m x的值: {} \033[0m".format(x))
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