import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

# 模型结构：输入(文本索引)->Embedding(pre-trained)->lstm->dropout->lstm->dropout->dense->输出(softmax)

# 预训练模型：包括word_to_index：{单词：索引}字典；index_to_word：{索引：单词}字典；word_to_vec_map：{单词：词向量}字典
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt')

# 文本预处理：文本->索引列表
def sentences_to_indices(X, word_to_index, max_len):
    """
    将文本数据转换一串索引组成的list
    
    输入:
    X -- 文本数据 (样本数, 1)
    word_to_index -- 单词:索引形式的字典
    max_len -- 数据集中最长的句子的长度
    
    输出:
    X_indices -- 索引矩阵 (样本数, 最大长度)
    """
    
    m = X.shape[0]                                   # 样本数
    
    # 零初始化
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # 遍历每个样本
        
        # 将文本句子转换为单词列表
        sentence_words =X[i].lower().split()
        
        for j, w in enumerate(sentence_words):					 # 遍历句中每个单词
            # i为样本索引，j为单词位置索引
            X_indices[i, j] = word_to_index[w]
				
    return X_indices

# Embedding层
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    读入预训练的权重，并创建Embedding层
    
    输入:
    word_to_vec_map -- 单词:预训练权重形式的字典
    word_to_index -- 单词:索引形式的字典

    输出:
    embedding_layer -- Keras层
    """
    
    vocab_len = len(word_to_index) + 1                  
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # 词向量维度
    
    # embedding矩阵的零初始化
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # 对于每个单词，将对应的权重放在索引位置上
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # 定义Keras层(不可训练)
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)

    # 将预训练结果赋给Keras层
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
	
# 模型
def model(input_shape, word_to_vec_map, word_to_index):
    """
    创建模型
    
    输入:
    input_shape -- 输入形式 (最大长度,)
    word_to_vec_map -- 单词:预训练权重形式的字典
    word_to_index -- 单词:索引形式的字典

    输出:
    model -- Keras模型
    """
    
    # 定义输入层
    inputs = Input(shape=input_shape,dtype='int32')
    
    # 预训练好的embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(inputs)     
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # LSTM
    X = LSTM(units=128, return_sequences=True)(embeddings) # 需要继续保留全部10个时间节点t
    # Dropout
    X = Dropout(0.5)(X)
    # LSTM
    X = LSTM(units=128,return_sequences=False)(X)		   # 只需要输入当个节点
    # Dropout
    X = Dropout(0.5)(X)
    # 输出层
    outputs = Dense(5, activation='softmax)(X)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    return model