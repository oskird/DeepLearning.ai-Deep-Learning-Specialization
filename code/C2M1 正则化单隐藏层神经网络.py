# L2正则化：损失函数的计算方式和反向传播时的梯度计算会有变化，但正向传播的过程没有变化
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    L2正则化的损失函数效果
    
    输入:
    A3 -- 当前参数下的模型预测值
    Y -- 标签
    parameters -- 当前模型参数
    
    输出:
    cost - 损失函数的值
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
	# 预测值与真实值的交叉熵损失值
    cross_entropy_cost = compute_cost(A3, Y) 
    # L2正则化的损失值
    L2_regularization_cost=lambd/(2*m)*np.sum(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    return cost
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    含有L2正则化的反向传播计算
    
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    cache -- 前向传播的计算结果
    lambd -- 正则化系数
    
    输出:
    gradients -- 字典，包含梯度计算结果
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
	# 第三层计算
    dZ3 = A3 - Y
    dW3=1/m*np.dot(dZ3, A2.T) + (lambd/m)*(W3) # 添加正则化因子后，dW的计算方式有变化
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
	# 第二层计算
    dZ2 = np.multiply(dA2, np.int64(A2 > 0)) # ReLU激活函数的梯度求法
    dW2=1/m*np.dot(dZ2, A1.T) + (lambd/m)*(W2)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
	# 第一层计算
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1=1/m*np.dot(dZ1, X.T) + (lambd/m)*(W1)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

# drop正则化：正向传播和反向传播过程因为有单元失活都会发生变化，损失函数的计算方式不变
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    含有dropout正则化的正向传播计算
    
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    parameters -- 参数字典
    keep_prob - 失活概率
    
    Returns:
    A3 -- 模型计算出的概率
    cache -- 保存中间计算过程
    """
    
    np.random.seed(1)
    
    # 取出参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # 第一层计算
    Z1 = np.dot(W1, X) + b1							  # 注意：计算Z1时，原始特征矩阵X一般是不添加失活效果的
    A1 = relu(Z1)
	# dropout过程
    D1 = np.random.rand(A1.shape[0],A1.shape[1])      # 1. 与A形式相同的随机矩阵
    D1 = D1 < keep_prob                               # 2. 使概率小于keep_prob的部分随机失活
    A1 = A1*D1                                        # 3. 对于失活的位置，将A中相应的位置记为0，相当于关闭了该节点
    A1 = A1/keep_prob                                 # 4. 对于没有失活的位置，用A/keep_prob扩大其权重，保持传播过程中的期望值不变
	# dropout过程结束
    # 第二层计算
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])      
    D2 = D2 < keep_prob                               
    A2 = A2*D2                                        
    A2 = A2/keep_prob                                 
    # 第三层计算
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    含有dropout正则化的反向传播计算
    
    Arguments:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    cache -- 前向传播的计算结果
    keep_prob - 失活概率
    
    Returns:
    gradients -- 字典，包含梯度计算结果
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
	# 第三层计算
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
	# dropout过程
    dA2 = dA2*D2              # 1. 反向传播中失活的位置应该和正向传播的相同
    dA2 = dA2/keep_prob       # 2. 对于没有失活的位置，用dA/keep_prob扩大其权重，保持传播过程中的期望值不变
    # dropout过程结束
	# 第二层计算
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1*D1               
    dA1 = dA1/keep_prob        
    ## 第一层计算
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    执行带有正则化方法的神经网络模型
    
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    learning_rate -- 学习速率
    num_iterations -- 迭代轮次
    print_cost -- 是否显示损失函数值
    lambd -- 正则化系数
    keep_prob - 节点保留率
    
    输出:
    parameters -- 可用于预测的最优化模型参数
    """
        
    grads = {}
    costs = []                            
    m = X.shape[1]                        
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # 初始化参数
    parameters = initialize_parameters(layers_dims)


    for i in range(0, num_iterations):

        # 正向传播计算预测结果
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters) # 当keep_prob=1时，相当于没有dropout效果
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob) # 当keep_prob!=1时，开启dropout
        
        # 损失函数
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd) # 如果lambda>0，相当于损失函数中添加了L2正则化
            
        # 反向传播
        assert(lambd==0 or keep_prob==1)    # 在此模型中，L2和dropout不可同时应用
		
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd) # 含有L2效果的反向传播
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob) # 含有dropout效果的反向传播
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每10000轮迭代，显示一次损失函数
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters