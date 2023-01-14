"""
-*- coding: UTF-8 -*-
@Time : 2023/1/14  21:18
@Description :
@Author : Huizhi XU
"""
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

BOS_TOKEN = "<bos>"  # beginning of sentence
EOS_TOKEN = "<eos>"  # end of sentence
PAD_TOKEN = "<pad>"  # 补齐标记


class NGramDataset(Dataset):
    """
    数据处理类
    """
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首、句尾标记符
            sentence = [self.bos] + sentence + [self.eos]

            if len(sentence) < context_size:
                continue

            for i in range(context_size, len(sentence)):
                # 模型输入：长度为context_size的上下文
                context = sentence[i - context_size:i]
                # 模型输出：当前词
                target = sentence[i]
                # 每个训练样本由（context, target）构成
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        """
        冬样本集合中构建批次的输入输出，转化为张量类型
        :param examples:
        :return:
        """

        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

        return inputs, targets
