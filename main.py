import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# arc_len = torch.randint(0,1,(5,0))
batch_size = 2
#词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8#词向量的的维度大小
#序列长度
max_src_seq_len = 5
max_tgt_seq_len = 5


src_len = torch.tensor([2,4],dtype=torch.int32)#源有两个句子，第一个句子2个词，第二个句子4个词
tgt_len = torch.tensor([4,3],dtype=torch.int32)#目标有两个句子，第一个句子4个词，第二个句子3个词

src_seq = torch.stack([
    F.pad(torch.randint(1, max_num_src_words, (int(L),)), (0,max_src_seq_len-L))
    for L in src_len
])
tgt_seq = torch.stack([
    F.pad(torch.randint(1, max_num_tgt_words, (int(L),)), (0,max_tgt_seq_len-L))
    for L in tgt_len
])
arc_embeding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embeding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

print(arc_embeding_table)
