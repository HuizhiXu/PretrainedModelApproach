"""
-*- coding: UTF-8 -*-
@Time : 2023/1/14  11:38
@Description :
@Author : Huizhi XU
"""

# 第一次运行需要下载数据
# import nltk
# nltk.download('reuters')
# nltk.download('punkt')


from nltk.corpus import reuters
from torch.utils.data import DataLoader

from static_word_embedding_models.vocab_mapping import Vocab

BOS_TOKEN = "<bos>"  # beginning of sentence
EOS_TOKEN = "<eos>"  # end of sentence
PAD_TOKEN = "<pad>"  # 补齐标记


def load_reuters():
    from nltk.corpus import reuters
    sentences = reuters.sents()
    text = [[word.lower() for word in sen] for sen in
            sentences]  # text 的格式[['asian', 'exporters', 'fear', 'damage'],['they', 'told', 'reuter', 'correspondents', 'in', 'asian']]
    vocab = Vocab.build(text, reserved_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]

    return corpus, vocab


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader

def save_pretrained(vocab, embeds, save_path):
    """
    保存词表以及训练得到的词向量
    :param vocab:
    :param embeds:
    :param save_path:
    :return:
    """
    with open(save_path, "w") as writer:
        # 记录词向量大小
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join([f"[x]" for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
