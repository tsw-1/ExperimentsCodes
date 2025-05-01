import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import accuracy_score

# 生成模拟数据
train_X = torch.randint(0, 30522, size=(100, 128))
train_Y = torch.randint(0, 2, size=(100,))
val_X = torch.randint(0, 30522, size=(20, 128))
val_Y = torch.randint(0, 2, size=(20,))
num_epochs=50
batch_size=100
learning_rate=0.0001
# 定义BERT分类器模型
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state # 取出BERT最后一层的输出
        pooled_output = bert_output.pooler_output # 取出BERT的pooler层的输出
        # 将两种输出拼接起来
        combined_output = torch.cat((last_hidden_state[:, 0, :], pooled_output), dim=1)
        # 经过全连接层和激活函数
        logits = self.classifier(combined_output)
        return logits

# 定义训练和评估函数
def train_and_evaluate(model, train_X, train_Y, val_X, val_Y, num_epochs, batch_size, learning_rate):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 将数据分成mini-batch
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        train_loss, train_acc = 0, 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, attention_mask=(batch_X > 0))
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            train_acc += accuracy_score(batch_Y.detach().numpy(), torch.argmax(outputs, dim=1).detach().numpy()) * batch_X.size(0)
        train_loss /= len(train_X)
        train_acc /= len(train_X)

        # 评估模型
        model.eval()
        val_outputs = model(val_X, attention_mask=(val_X > 0))
        val_loss = criterion(val_outputs, val_Y).item()
        val_acc = accuracy_score(val_Y.detach().numpy(), torch.argmax(val_outputs, dim=1).detach().numpy())

        print('Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

if __name__ == '__main__':
        model=BertClassifier(2)
        train_and_evaluate(model, train_X, train_Y, val_X, val_Y, num_epochs, batch_size, learning_rate)
