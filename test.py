import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

arc_len = torch.randint(0, 1, (5, 0))
batch_size = 2
#词表大小
max_num_src_words = 2
max_num_tgt_words = 8

#序列长度
# max_src_seq_len = 5
# max_tgt_seq_len = 5
#
#
# src_len = torch.tensor([2,4],dtype=torch.int32)
# tgt_len = torch.tensor([4,3],dtype=torch.int32)
#
# src_seq1 = [torch.randint(1,max_num_src_words,(L,))
#             for L in src_len]
# src_seq2 = [F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len-L))
#             for L in src_len]
# src_seq3 =[torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len-L)),0)
#             for L in src_len]
# src_seq4 =torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len-L)),0)
#             for L in src_len])
# #将src_seq4优化为以下写法
# src_seq5 = torch.stack([
#     F.pad(torch.randint(1, max_num_src_words, (int(L),)), (0,max_src_seq_len-L))
#     for L in src_len
# ])
# print(src_seq1,'----------')
# print(src_seq2,'----------')
# print(src_seq3,'----------')
# print(src_seq4,'----------')
# print(src_seq5,'----------')

#以下为执行结果
# [tensor([1, 1]), tensor([1, 1, 1, 1])] ----------
# [tensor([1, 1, 0, 0, 0]), tensor([1, 1, 1, 1, 0])] ----------
# [tensor([[1, 1, 0, 0, 0]]), tensor([[1, 1, 1, 1, 0]])] ----------
# tensor([[1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 0]]) ----------
# tensor([[1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 0]]) ----------

#-----------------------------------
# embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)
# input_ids = torch.tensor([1, 3, 5])
# out = embedding(input_ids)
# print(embedding.weight)
# print(out)

# a= torch.arange(7).view(1,-1)
# print(a)
a= torch.zeros(5,3)
emb = nn.Embedding(5, 3)
print(emb.weight)
emb.weight = nn.Parameter(a)
print(emb.weight)
print(emb.weight.shape)   # [5, 3]

idx = torch.tensor([0, 2, 4])
out = emb(idx)
print(out)
print(out.size())
