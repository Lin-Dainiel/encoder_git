import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
# arc_len = torch.randint(0,1,(5,0))
#batch_size代表一次处理的句子数量，比如该模型一次可以处理2个句子(batch_size)，每个句子5个词(max_src_seq_len)，每个词的维度为8(model_dim)
batch_size = 2

#词表大小
max_num_src_words = 8
max_num_tgt_words = 8

#词向量的的维度大小
model_dim = 8
#序列长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_pos_seq_len = 5

src_len = torch.tensor([2,4],dtype=torch.int32)#源有两个句子，第一个句子2个词，第二个句子4个词
tgt_len = torch.tensor([4,3],dtype=torch.int32)#目标有两个句子，第一个句子4个词，第二个句子3个词

#根据句子，生成各个词在词表中的序列，也就是单词索引
src_seq = torch.stack([
    F.pad(torch.randint(1, max_num_src_words, (int(L),)), (0,max_src_seq_len-L))
    for L in src_len
])
tgt_seq = torch.stack([
    F.pad(torch.randint(1, max_num_tgt_words, (int(L),)), (0,max_tgt_seq_len-L))
    for L in tgt_len
])

#以下为构造words embedding
arc_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)#词嵌入表，行数为词表数量+1，列数为每个词的维度，+1留给了特殊token
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)
arc_embedding = arc_embedding_table(src_seq)#根据各个词的序列，拿到词嵌入表中对应的向量
tgt_embedding = tgt_embedding_table(tgt_seq)

#以下为构造positon embedding
pos_mat = torch.arange(max_pos_seq_len).view(-1,1)#8x1张量
i_mat = torch.exp(torch.arange(0,model_dim,2)*-math.log(10000)/model_dim)#1x4张量
pe_embedding_table = torch.zeros(max_pos_seq_len, model_dim)
pe_embedding_table[:,0::2] = torch.sin(i_mat*pos_mat)#张量的广播机制，i_mat*pos_mat得到一个8x4的矩阵
pe_embedding_table[:,1::2] = torch.cos(i_mat*pos_mat)

pe_embedding = nn.Embedding(max_pos_seq_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding, requires_grad=False)
arc_pos_embedding_= pe_embedding(max_pos_seq_len, model_dim)

print(torch.sin(i_mat*pos_mat))
print(torch.sin(pos_mat*i_mat))
print(pe_embedding_table)