import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 构建Embedding类实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self,vocab, d_model):
        super(Embeddings, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        #将参数传入类中
        self.d_model= d_model

    def forward(self,x):
        # x代表输入进模型的文本
        return self.lut(x) * math.sqrt(self.d_model)
    

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100,23,4,58], [765,998,54,1]]))
emb = Embeddings(vocab, d_model)
result = emb(x)
# print(result)
# print(result.shape)

# m = nn.Dropout(p=0.2)
# input1 = torch.randn(4,5)
# output1 = m(input1)
# print(output1)

# x = torch.tensor([1,2,3,4])
# y = torch.unsqueeze(x,0)
# print(y)
# z = torch.unsqueeze(x,1)
# print(z)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model:词嵌入的维度
        # dropout；dropout层置零比率
        # max_len：代表句子最大长度
        super(PositionalEncoding, self).__init__()

        # 实例化dropout层
        self.dropout = nn.Dropout(dropout)

        # 初始化位置编码矩阵，大小是（max_len * d_model）
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变化矩阵div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 将前面定义的变换矩阵进行奇数偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)

        # 将位置编码器注册成模型的buffer， 这个buffer不是模型的参数，不跟随优化器更新
        # 注册成buffer后，可以在模型保存后重新加载的时候，将位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)

    def forward(self,x):
        # x：文本序列的词嵌入表示
        # pe编码太长了，将第二个维度max_len缩小成x的句子长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
d_model = 512
dropout = 0.1
max_len = 60

x = result
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
# print(pe_result)
# print(pe_result.shape)

plt.figure(figsize=(15,5))
# 实例化PositionalEncoding对象，词嵌入维度20，置零比率为0
pe = PositionalEncoding(20,0)
# 向pe中传入全零初始化的x，相当于展示pe
y = pe(Variable(torch.zeros(1,100,20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])




# print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=-1))

# print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=0))

# print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], k=1))

# 构建掩码张量的函数
def subsequent_mask(size):
    # size代表掩码张量后两个维度
    attn_shape = (1, size, size)

    # 使用np.one()构建一个全一的张量，然后使用np.triu()形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 是这个三角矩阵反转
    return torch.from_numpy(1-subsequent_mask)

# size =5
# sm = subsequent_mask(size)
# plt.figure(figsize=(5,5))
# plt.imshow(sm[0])


# x = Variable(torch.randn(5,5))
# print(x)

# mask = Variable(torch.zeros(5,5))
# print(mask)

# y = x.masked_fill(mask == 0, -1e9)
# print(y)


def attention(query, key, value, mask=None, dropout=None):
    # query key value 代表注意力三个输入张量
    # mask：掩码张量
    # dropout：传入的Dropout实例化对象
    # 首先将query最后一个维度提取出来，代表词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力计算公式，将query和key的转置矩阵相乘，然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        #利用masked_fill()方法，将掩码张量和0进行位置的一一比较，如果等于0，替换成一个非常小的数值
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 对scores最后一个维度上进行softmax操作
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后一步完成p_attn和value张良的乘法，并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result    #query = key = value 称作自注意力机制
mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value, mask=mask)
# print(attn)
# print(attn.shape)
# print(p_attn)
# print(p_attn.shape)

# x = torch.randn(4,4)
# print(x.size())
# y = x.view(16)
# print(y.size())
# z = x.view(-1, 8)
# print(z.size())

# 实现克隆函数，因为在多头注意力机制下，要用到多个结构的线性层
# 需要使用clone函数，将他们一同初始化到一个网络层列表对象中
def clones(module, N):
    # module:代表克隆的目标网络层
    # N:将module克隆几个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制的类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head：代表几个头的参数
        # embedding_dim:代表词嵌入的维度
        #  dropout:置零的比率
        super(MultiHeadedAttention, self).__init__()

        # head需要整除embedding_dim
        assert embedding_dim % head == 0

        # 得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        # 获得线性层，要获得四个，分别是Q，K，V，以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张良
        self.attn = None

        # 初始化dropoutduixiang 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 使用squeeze将掩码张量进行维度扩充，代表多头中的第n个头
            mask = mask.unsqueeze(1)

        # 得到batchsize
        batch_size = query.size(0)

        # 首先使用zip将网络层和输入连接到一起，模型的输出利用view和transpose进行围堵和形状的改变
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入到注意力层
        x, self.attn = attention(query, key, value, mask = mask, dropout=self.dropout)
        # 得到每个头的计算结果是四维张量，需要进行形状的转换
        # 前面已经将1，2维度进行了转置，这里要重新转置回来
        # 注意：经历了transpose()方法后，必须使用contiguous方法，不然无法使用view()方法
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后将x输入到线性层列表中最后一个线性层，得到最终的多头注意力输出
        return self.linears[-1](x)

# 实例化若干参数
head = 8
embedding_dim = 512
dropout = 0.2

# 若干输入参数的初始化
query = key = value = pe_result

mask = Variable(torch.zeros(2,4,4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)


# 构建前馈全连接网络类
class PositionwiseForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model:代表词嵌入维度，同时也是两个线性层的输入和输出维度
        # d_ff:代表第一个线性层的输出维度，和第二个线性层的输入维度
        super(PositionwiseForward, self).__init__()

        # 定义两层的全连接线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x：代表上一层的输出
        # 首先将x送入第一个线性层，然后经relu函数激活，在经过dropout处理
        # 最后送入第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))
    
d_model= 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseForward(d_model, d_ff, dropout)
ff_result = ff(x)
# print(ff_result)
# print(ff_result.shape)



# 构建规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features：代表词嵌入的维度
        # eps一个足够小的正数，用老在规范化计算公式的分母中，防止除0
        super(LayerNorm, self).__init__()

        # 初始化两个参数张良a2,b2，用于对结果做化操作
        # 将启用nn.Parameter进行封装，代表他们也是抹胸中的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        # x:代表上一层网络的输出
        # 首先对x最后一个维度求均值操作，同时保持输出维度和输入维度一致
        mean = x.mean(-1, keepdim = True)
        # 接着对x最后一个维度商丘标准差的操作，同时保持输出维度和输入维度一致
        std = x.std(-1, keepdim = True)
        # 按规范化公式计算并返回
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
    
features = d_model = 512
eps = 1e-6

x = ff_result
ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)


# 构建子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size：次词嵌入维度
        super(SublayerConnection, self).__init__()
        # 实例化一个规范层的对象
        self.norm = LayerNorm(size)
        # 实例化dropout对象
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        # x：代表上一层传入的张量
        # sublayer：该子层连接中，子层函数
        # 首先将x进行规范化，然后送入子层函数中处理，处理结果进入dropout，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))
    
size = d_model =512
head = 8
dropout = 0.2
x = pe_result
mask = Variable(torch.zeros(2,4,4))
self_attn = MultiHeadedAttention(head, d_model)

sublayer = lambda x: self_attn(x,x,x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)


# 构建编码器层的类
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size;代表词嵌入维度
        # self_attn：代表传入的多头自注意力子层的实例化对象
        # feed_forward：代表前馈全连接层的实例化对像
        super(EncoderLayer, self).__init__()

        # 将两个实例化对象和参数传入类中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        # 编码器中有两个子层连接结构，用clons函数操作
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # x:代表上一层的传入张量
        # 首先让x进入第一个子层连接结构，内部包含多头注意力机制子层
        # 再让张良进入第二个子层连接结构，其中包括前馈全连接网络
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)
    
size = d_model = 512
head = 8
d_ff = 64
x = pe_result
dropout = 0.2

self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(2,4,4))

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x,mask)
# print(el_result)
# print(el_result.shape)


# 构建编码器类
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer：编码器层
        # N：代表编码器有几层
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layer中
        self.layers = clones(layer, N)
        # 初始化一个规范化层,作用在整个编码器的后面
        self.norm = LayerNorm(layer.size)

    def forward(self,x, mask):
        # x:代表上一层输出的丈量
        # 让x依次经历N个编码器的处理，最后再经过规范化层输出就好了
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
size = d_model = 512
d_ff = 64
head = 8
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = Variable(torch.zeros(2,4,4))

en = Encoder(layer, N)
en_result = en(x,mask)
print(en_result)
print(en_result.shape)