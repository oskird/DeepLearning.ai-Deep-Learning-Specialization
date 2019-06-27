from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

# 方法概述：
# 注意力机制模型是seq2seq的一种，因此和常规的seq2seq一样，包括了两个循环神经网络部分，一个针对输入X的序列(一般是双向的循环神经网络)，一个针对输出Y的序列。
# 注意力机制是给X循环神经网络的输出结果赋予一个权重α，再对Y序列的节点作用。α的计算过程：由前一层的s和本层的a共同得到e，再对e进行softmax得到权重α。
# 得到权重α后，与输出a相乘得到c，为输入到Y序列的注意力计算结果；c与前一层的s共同作用得到Y的输出结果；再计算出本层的s，用来计算下一层的注意力权重和输出。

# 重要参数
Tx = 30               # 输入x的时间节点数
Ty = 10               # 输出y的时间节点数
Lx = 37               # 输入x的词库长度
Ly = 11               # 输出y的词库长度
n_a = 32			  # a的隐藏节点数
n_s = 64			  # s的隐藏节点数

# 单步注意力计算
repeator = RepeatVector(Tx) 								# 将向量重复Tx次：(m, n) -> (m, Tx, n)
concatenator = Concatenate(axis=-1)							# 将向量合并
densor1 = Dense(10, activation = "tanh")					# 全连接层，沿着-1轴：(m, Tx, n) -> (m, Tx, 10)
densor2 = Dense(1, activation = "relu")						# 全连接层，沿着-1轴：(m, Tx, n) -> (m, Tx, 1)
activator = Activation(softmax, name='attention_weights')   # 激活层，沿着1轴：(m, Tx, 1) -> (m, Tx, 1)
dotor = Dot(axes = 1)										# 点乘法，用于向量计算，沿着1轴：(m, Tx, 1) & (m, Tx, n) -> (m, 1, n)
def one_step_attention(a, s_prev):
    """
    半步注意力机制
    
    输入:
    a -- 本层序列X的双向循环神经网络计算结果 (样本数, 时间节点数, 2*a隐藏节点数)
    s_prev -- 前一层序列Y的post-attention计算结果 (m, s隐藏节点数)
    
    输出:
    context -- 注意力计算结果c
    """
    
    # 将上一层的a重复Tx次 
    s_prev = repeator(s_prev)
    # 合并a和s
    concat = concatenator([a,s_prev])
    # 全连接层1
    e = densor1(concat)
    # 全连接层2：计算出每个节点的e
    energies = densor2(e)
    # softmax激活来计算权重α
    alphas = activator(energies)
    # 点乘计算c
    context = dotor([alphas,a])
    return context

# 完整的含注意力机制的seq2seq模型
post_activation_LSTM_cell = LSTM(n_s, return_state = True)  		# Y序列的循环网络层LSTM
output_layer = Dense(len(machine_vocab), activation=softmax)  		# Y序列的输出层 
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    输入:
    Tx -- 输入x的时间节点数
    Ty -- 输出y的时间节点数
    n_a -- a的隐藏节点数
    n_s -- s的隐藏节点数
    Lx  --  输入x的词库长度
    Ly  --  输出y的词库长度

    输出:
    model -- 注意力机制魔性的LSTM实例
    """
    
    # 设置输入层
    X = Input(shape=(Tx, human_vocab_size))	
    s0 = Input(shape=(n_s,), name='s0')				# s0初始化
    c0 = Input(shape=(n_s,), name='c0')				# c0初始化
    s = s0
    c = c0
    
    # 输出
    outputs = []
    
    # X序列一般是双向的LSTM(也叫作pre-attention)
    a = Bidirectional(LSTM(n_a, return_sequences = True))(X)
    
    # 对于每个Y序列节点进行遍历
    for t in range(Ty):
    
        # 注意力计算结果
        context = one_step_attention(a,s)
        
        # Y序列的LSTM(也叫作post-attention)
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])       # 保留s和c的更新结果
        
        # 通过全连接层和softmax激活计算输出
        out = output_layer(s)
        
        # 将该时间节点的最终计算加入输出结果
        outputs.append(out)
    
    # 定义输入和输出，创建模型
    model = Model(inputs = [X,s0,c0], outputs = outputs)

    return model