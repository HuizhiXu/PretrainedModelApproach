"""
-*- coding: UTF-8 -*-
@Time : 2023/1/14  13:49
@Description : Vocab类主要用于token和索引的相互映射
@Author : Huizhi XU
"""

from collections import defaultdict
from typing import TypeVar, Generic, Optional
from pydantic import BaseModel


class Vocab():
    """
    Vocab类主要用于token和索引的相互映射
    """

    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens += ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        """
        构建词表
        :param text: 输入文本
        :param min_freq: 出现的最小次数
        :param reserved_tokens:
        :return:
        """
        token_freqs = defaultdict(int)  # 用defaultdict防止KeyError
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items()
                        if freq >= min_freq and token != "<unk>"]

        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        """
        将token转换为索引
        :param tokens: 语言符号
        :return:
        """
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        """
        将索引转化为token
        :param indices: 索引
        :return: 索引对应的token
        """
        return [self.idx_to_token[index] for index in indices]
