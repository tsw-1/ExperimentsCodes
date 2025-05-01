class BERTHierClassifierTransAbs(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        # 初始化函数，定义模型结构和参数
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type

        # 帖子编码器，使用预训练的BERT模型进行帖子的编码
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            # 冻结BERT模型的参数，使其在训练过程中不被更新
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False

        # 获取BERT模型的隐藏层维度
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts

        # 位置编码矩阵，为每个帖子的位置提供编码信息，以便Transformer编码器能够处理序列中不同位置的信息
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')

        # 用户编码器，使用多层的Transformer Encoder进行用户的编码
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)

        # 线性层，用于计算用户编码器输出中每个帖子的权重，将所有帖子的信息合并为单个用户向量
        self.attn_ff = nn.Linear(self.hidden_dim, 1)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)

        # 分类器，将用户编码的输出映射到一个单一的输出值，用于分类任务
        self.clf = nn.Linear(self.hidden_dim, 1)  # self.clf 是一个 nn.Linear 的实例

    def forward(self, batch, **kwargs):
        '''
        前向传播函数，计算模型的输出
        feats是一个列表，其中包含每个用户的特征向量。
        每个特征向量是一个加权平均后的帖子向量，权重由self.attn_ff计算。
        feats的形状为(batch_size, hidden_size)，其中batch_size是一批用户的数量，hidden_size是特征向量的维度。
        '''
        feats = []  # 用于存储每个用户的特征向量
        attn_scores = []  # 用于存储每个用户的注意力分数

        for user_feats in batch:
            # 使用BERT模型编码帖子
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])

            # 从BERT的输出中选择帖子的表示
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]  # 选择每个序列的第一个词的表示
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)  # 对帖子表示进行平均池化

            # 为帖子添加位置编码
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)

            # 使用Transformer编码器对帖子进行编码
            x = self.user_encoder(x).squeeze(1)  # [num_posts, hidden_size]

            # 计算每个帖子在用户级别上的重要性分数
            attn_score = torch.softmax(self.attn_ff(x).squeeze(-1), -1)  # torch.Size([num_posts, ])
            
            # 计算加权平均后的用户特征向量
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)

        # 将特征列表转换为张量
        feats = torch.stack(feats)
        
        # 对特征向量进行Dropout操作，防止过拟合
        x = self.dropout(feats)
        
        # 使用分类器映射到单一的输出值
        logits = self.clf(x).squeeze(-1)
        
        # 返回模型的输出
        return logits, attn_scores
