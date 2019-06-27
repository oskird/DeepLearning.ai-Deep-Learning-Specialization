import numpy as np

# RNN单步计算
def rnn_cell_forward(xt, a_prev, parameters):
    """
    循环神经网络的一步计算

    输入:
    xt -- 时间t的特征矩阵 (特征数, 样本数).
    a_prev -- 时间t-1的计算结果矩阵 (隐藏节点数, 样本数)
    parameters -- 参数字段:
                        Wax -- 与特征矩阵相乘的参数矩阵 (隐藏节点数, 样本数)
                        Waa -- 与前一时间点的计算结果矩阵相乘的参数矩阵 (隐藏节点数, 隐藏节点数)
                        Wya -- 与输出矩阵相乘的参数矩阵 (输出节点数, 隐藏节点数)
                        ba --  计算结果的偏置项 (隐藏节点数, 1)
                        by --  输出的偏置项 (输出节点数, 1)
    输出:
    a_next -- 时间t的计算结果矩阵 (隐藏节点数, 样本数)
    yt_pred -- 时间t的输出矩阵 (输出节点数, 样本数)
    cache -- 缓存字典，包括a_next, a_prev, xt, parameters
    """
    
    # 提取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # 计算时间t的结果
    a_next = np.tanh(np.dot(Waa, a_prev)+np.dot(Wax, xt)+ba)
    # 计算时间t的输出
    yt_pred = softmax(np.dot(Wya,a_next) + by)   
    
    # 保存中间计算过程
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache
	
# RNN序列计算
def rnn_forward(x, a0, parameters):
    """
    循环神经网络序列计算

    输入:
    x -- 特征矩阵 (特征数, 样本数, 序列长度).
    a0 -- 初始的计算结果矩阵 (隐藏节点数, 样本数)
    parameters -- 参数字段:
                        Wax -- 与特征矩阵相乘的参数矩阵 (隐藏节点数, 样本数)
                        Waa -- 与前一时间点的计算结果矩阵相乘的参数矩阵 (隐藏节点数, 隐藏节点数)
                        Wya -- 与输出矩阵相乘的参数矩阵 (输出节点数, 隐藏节点数)
                        ba --  计算结果的偏置项 (隐藏节点数, 1)
                        by --  输出的偏置项 (输出节点数, 1)

    输出:
    a -- 计算结果矩阵 (隐藏节点数, 样本数, 序列长度)
    y_pred -- 输出矩阵 (输出节点数, 样本数, 序列长度)
    caches -- 缓存字典，包括每一时间点缓存结果和特征矩阵
    """
    
    # 初始缓存字典
    caches = []
    
    # 获取维度
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # 初始a和y
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y,m,T_x))
	
    # 起始a设为a0
    a_next = a0
    
    # 序列循环
    for t in range(T_x):
        # 计算时间t节点上的结果
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # 保存a和y
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        # 保留中间计算结果
        caches.append(cache)
    
    caches = (caches, x)
    
    return a, y_pred, caches