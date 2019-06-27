# 填充
def zero_pad(X, pad):
    """
    用0对矩阵进行填充。由于填充是对图像进行的，因此仅填充高和宽的两个维度，不填充样本和信道维度。
    
    输入:
    X -- 矩阵 (样本数，高, 宽, 信道数)
    pad -- 填充数量
    
    输出:
    X_pad -- 填充后的矩阵 (样本数，高+2*pad, 宽+2*pad, 信道数)
    """
    
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    
    return X_pad
	
# 单层卷积计算
def conv_single_step(a_slice_prev, W, b):
    """
    使用过滤器进行一次卷积计算。
    
    输入:
    a_slice_prev -- 图像数据的一部分 (过滤器大小, 过滤器大小, 前一层的信道数)
    W -- 权重矩阵 (过滤器大小, 过滤器大小, 前一层的信道数)
    b -- 偏置项，对于前一层的每一个信道，偏置项都是相同的 (1, 1, 1)
    
    Returns:
    Z -- 卷积计算结果，标量
    """

    # 计算乘积
    s = a_slice_prev*W
    # 求和
    Z = np.sum(s)
    # 添加偏置项
    Z = Z+np.squeeze(b)

    return Z
	
# 卷积层前向传播
def conv_forward(A_prev, W, b, hparameters):
    """
    执行一次卷积层前向传播
    
    Arguments:
    A_prev -- 前一层计算结果 (样本数, 前一层的高, 前一层的宽, 前一层的信道数)
    W -- 权重矩阵 (过滤器大小, 过滤器大小, 前一层的信道数，本层的信道数)
    b -- 偏置项, 本层的每一个信道偏置项不一样 (过滤器大小, 过滤器大小, 前一层的信道数，本层的信道数)
    hparameters -- 超参数，包括步长和填充大小
        
    Returns:
    Z -- 卷积层输出 (样本数, 高, 宽, 本层信道数)
	cache -- 保存中间计算结果(供反向传播使用)
    """
    
    # 获取(样本数, 前一层的高, 前一层的宽, 前一层的信道数)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # 获取过滤器
    (f, f, n_C_prev, n_C) = W.shape
    
    # 获取步长和填充大小
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # 计算本层输出的高和宽
    n_H = int(np.floor((n_H_prev-f+2*pad)/stride+1))
    n_W = int(np.floor((n_W_prev-f+2*pad)/stride+1))
    
    # 输出矩阵
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # 填充前层输出结果
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                                 # 样本遍历
        a_prev_pad = A_prev_pad[i]                     
        for h in range(n_H):                           # 高遍历
            for w in range(n_W):                       # 宽遍历
                for c in range(n_C):                   # 信道遍历
                    
                    # 找到与过滤器进行计算的slice
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # 进行卷积计算，并找到在本层的对应位置
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c],b[:,:,:,c])
    
    # 本层计算形状检查
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # 保存中间计算结果
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

# 池化层
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    执行一次池化层前向传播
    
    Arguments:
    A_prev -- 前一层计算结果 (样本数, 前一层的高, 前一层的宽, 前一层的信道数)
    hparameters -- 超参数，包括池化过滤器大小和步长
    mode -- 池化模式，包括max和average
    
    Returns:
    A -- 池化层输出 (样本数, 高, 宽, 本层信道数)
    cache -- 保存中间计算结果(供反向传播使用)
    """
    
    # 获取(样本数, 前一层的高, 前一层的宽, 前一层的信道数)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # 获取池化过滤器大小和步长
    f = hparameters["f"]
    stride = hparameters["stride"]
    
   # 计算本层输出的高、宽和信道数
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # 输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                                 # 样本遍历
        for h in range(n_H):                           # 高遍历
            for w in range(n_W):                       # 宽遍历
                for c in range(n_C):                   # 信道遍历
                    
                    # 找到与过滤器进行计算的slice
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    a_slice_prev = A_prev[i,vert_start:vert_end, horiz_start:horiz_end,c]
                    
                    # 进行池化计算，并找到在本层的对应位置
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
    
    # 本层计算形状检查
    assert(A.shape == (m, n_H, n_W, n_C))
	
    # 保存中间计算结果
    cache = (A_prev, hparameters)
    
    return A, cache