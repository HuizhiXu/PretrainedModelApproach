"""
-*- coding: UTF-8 -*-
@Time : 2023/1/14  22:31
@Description :
@Author : Huizhi XU
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import load_reuters, get_loader, save_pretrained
from datasets import NGramDataset
from models import FeedForward

embedding_dim = 128  # 词向量维度
hidden_dim = 256  # 隐含层维度
batch_size = 1024  # 批次大小
context_size = 3  # 输入上下文长度
num_epoch = 10  # 训练迭代次数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 读取文本数据，构建训练数据集
corpus, vocab = load_reuters()
dataset = NGramDataset(corpus, vocab, context_size)
data_loader = get_loader(dataset, batch_size)

# 负对数似然损失函数
nll_loss = nn.NLLLoss()

# 构建模型
model  = FeedForward(len(vocab),embedding_dim, context_size, hidden_dim)
model.to(device)

# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr= 0.0001)

model.train()
total_losses = []
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")
    total_losses.append(total_loss)

save_pretrained(vocab,model.embeddings.weight.data, "feedforward.vec")