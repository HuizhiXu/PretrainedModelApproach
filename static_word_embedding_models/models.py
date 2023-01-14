"""
-*- coding: UTF-8 -*-
@Time : 2023/1/14  22:00
@Description :
@Author : Huizhi XU
"""
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    前馈神经网络：包含词向量层、词向量到隐含层、隐含层到输出层
    """

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        """

        :param vocab_size: vocab的长度，可用len(vocab)计算
        :param embedding_dim:
        :param context_size:
        :param hidden_dim:
        """
        super(FeedForward, self).__init__()

        # 词向量层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 词向量到隐含层
        self.linear1 = nn.Linear(context_size*embedding_dim, hidden_dim)
        # 隐含层到输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 使用ReLU激活函数
        self.activate = F.relu

    def forward(self, inputs):
        """
        将输入文本序列映射成词向量，并通过view函数对映射后的词向量序列组成的三维张量进行重构，对词向量进行拼接
        :param inputs:
        :return:
        """

        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)
        hidden = self.activate(self.linear1(embeds))
        output = self.linear2(hidden)

        # 根据输出层logits进行概率分布并取对数，以便计算对数似然
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
