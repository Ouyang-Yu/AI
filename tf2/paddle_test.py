import numpy as np
import paddle

# 准备数据
# 加载IMDB数据
imdb_train = paddle.text.datasets.Imdb(mode='train')  # 训练数据集
imdb_test = paddle.text.datasets.Imdb(mode='test')  # 测试数据集
# 获取字典
word_dict = imdb_train.word_idx
# 在字典中增加一个<pad>字符串
word_dict['<pad>'] = len(word_dict)
# 参数设定
vocab_size = len(word_dict)
embedding_size = 256
hidden_size = 256
n_layers = 2
dropout = 0.5
seq_len = 200
batch_size = 64
epochs = 10
pad_id = word_dict['<pad>']


# 每个样本的单词数量不一样，用Padding使得每个样本输入大小为seq_len
def padding(dataset):
    padded_sents = []
    labels = []
    for batch_id, data in enumerate(dataset):
        sent, label = data[0].astype('int64'), data[1].astype('int64')
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int64')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels)


train_x, train_y = padding(imdb_train)
test_x, test_y = padding(imdb_test)


class IMDBDataset(paddle.io.Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.sents)


train_dataset = IMDBDataset(train_x, train_y)
test_dataset = IMDBDataset(test_x, test_y)

train_loader = paddle.io.DataLoader(train_dataset,
                                    return_list=True,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset,
                                   return_list=True,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   drop_last=True)


# 构建模型
class LSTM(paddle.nn.Layer):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = paddle.nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = paddle.nn.LSTM(embedding_size,
                                         hidden_size,
                                         num_layers=n_layers,
                                         direction='bidirectional',
                                         dropout=dropout)
        self.linear = paddle.nn.Linear(in_features=hidden_size * 2, out_features=2)
        self.dropout = paddle.nn.Dropout(dropout)

    def forward(self, text):
        # 输入text形状大小为[batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded形状大小为[batch_size, seq_len, embedding_size]
        output, (hidden, cell) = self.lstm_layer(embedded)
        # output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        # hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        # 把前向的hidden与后向的hidden合并在一起
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        hidden = self.dropout(hidden)
        # hidden形状大小为[batch_size, hidden_size * num_directions]
        return self.linear(hidden)


# 以下使用PaddlePaddle2.0高层API进行训练与评估
# 封装模型
model = paddle.Model(LSTM())  # 用Model封装lstm模型

# 配置模型优化器、损失函数、评估函数
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

# 模型训练与评估
model.fit(train_loader,
          test_loader,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1)
