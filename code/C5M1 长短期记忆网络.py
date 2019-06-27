import numpy as np

# LSTM单步计算
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    LSTM的一步计算

    输入:
    xt -- 时间t的特征矩阵 (特征数, 样本数).
    a_prev -- 时间t-1的计算结果矩阵 (隐藏节点数, 样本数)
    c_prev -- 时间t-1的记忆矩阵 (隐藏节点数, 样本数)
    parameters -- 参数字段:
                        Wf -- 遗忘门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bf -- 遗忘门偏置项 (隐藏节点数, 1)
                        Wi -- 输入门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bi -- 输入门偏置项 (隐藏节点数, 1)
                        Wc -- 记忆单元矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bc -- 记忆单元偏置项 (隐藏节点数, 1)
                        Wo -- 输出门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bo -- 输出门偏置项 (隐藏节点数, 1)
                        Wy -- 输出计算矩阵权重 (输出节点数, 隐藏节点数)
                        by -- 输出计算矩阵偏置项 (输出节点数, 1)
                        
    输出:
    a_next -- 时间t的计算结果矩阵 (隐藏节点数, 样本数)
    c_next -- 时间t的记忆矩阵 (隐藏节点数, 样本数)
    yt_pred -- 时间t的输出矩阵 (输出节点数, 样本数)
    缓存字典 -- 缓存字典，包括a_next, c_next, a_prev, c_prev, xt, parameters
    """

    # 提取参数
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    
    # 提取维度
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 合并a和x
    concat = np.zeros((n_x + n_a,m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # 计算过程
    ft = sigmoid(np.dot(Wf, concat)+bf)  # 遗忘门
    it = sigmoid(np.dot(Wi, concat)+bi)  # 输入门
    cct = np.tanh(np.dot(Wc, concat)+bc) # t时间节点记忆单元计算
    c_next = it*cct+ft*c_prev 			 # 利用遗忘和输入门控制记忆结果
    ot = sigmoid(np.dot(Wo, concat)+bo)  # 输出门
    a_next = ot*np.tanh(c_next)			 # t时间节点计算结果
    
    # 计算输出值
    yt_pred = softmax(np.dot(Wy, a_next)+by)

    # 保存中间计算过程
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache
	
# LSTM序列计算
def lstm_forward(x, a0, parameters):
    """
    LSTM序列计算

    输入:
    x -- 特征矩阵 (特征数, 样本数, 序列长度).
    a0 -- 初始的计算结果矩阵 (隐藏节点数, 样本数)
    parameters -- 参数字段:
                        Wf -- 遗忘门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bf -- 遗忘门偏置项 (隐藏节点数, 1)
                        Wi -- 输入门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bi -- 输入门偏置项 (隐藏节点数, 1)
                        Wc -- 记忆单元矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bc -- 记忆单元偏置项 (隐藏节点数, 1)
                        Wo -- 输出门矩阵权重 (隐藏节点数, 隐藏节点数 + 样本数)
                        bo -- 输出门偏置项 (隐藏节点数, 1)
                        Wy -- 输出计算矩阵权重 (输出节点数, 隐藏节点数)
                        by -- 输出计算矩阵偏置项 (输出节点数, 1)
                        
    输出:
    a -- 计算结果矩阵 (隐藏节点数, 样本数, 序列长度)
    y_pred -- 输出矩阵 (输出节点数, 样本数, 序列长度)
    caches -- 缓存字典，包括每一时间点缓存结果和特征矩阵
    """

    # 初始缓存字典
    caches = []
    
    # 获取维度
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # 初始a、c和y
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # 设置起始a和c
    a_next = a0
    c_next = np.zeros((n_a,m))
    
    # 序列循环
    for t in range(T_x):
        # 计算时间t节点上的结果
        a_next, c_next, yt, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        # 保存a、c和y
        a[:,:,t] = a_next
        y[:,:,t] = yt
        c[:,:,t]  = c_next
        # 保留中间计算结果
        caches.append(cache)
		
    caches = (caches, x)

    return a, y, c, caches