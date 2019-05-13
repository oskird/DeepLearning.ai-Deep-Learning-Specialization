# 参数初始化
def initialize_parameters_deep(layer_dims):
    """
    参数初始化
    输入：
	layer_dims -- 层数列表，包含神经网络每一层的节点个数。L[0]表示输入层节点数，L[len(L)-1]表示输出层节点数，列表的长度为节点数+1
    
    输出:
    parameters -- 初始化参数"W1", "b1", ..., "WL", "bL":
                    Wl -- 每层的参数矩阵 (当前层节点数, 前一层节点数)
                    bl -- 偏置项向量 (当前层节点数, 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # 神经网络层数
	
	# 对于每层，W进行随机初始化，b进行0初始化
    for l in range(1, L):
        parameters['W' + str(l)]=np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)]=np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
	
# 线性前向传播计算
def linear_forward(A, W, b):
    """
    执行一次前向传播计算

    输入:
    A -- 上一层激活后的计算结果 (前一层的节点数, 样本数)
    W -- 参数矩阵 (当前层节点数, 前一层的节点数)
    b -- 偏置项向量	(当前层节点数, 1)

    输出:
    Z -- 当前层的线性计算(激活前)结果 (当前层的节点数, 样本数)
    cache -- 储存本次A/W/b的字典
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z=np.dot(W,A)+b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
	
# 激活函数
def sigmoid(Z):
    """
    执行sigmoid激活
    
    输入:
    Z -- 当前层的线性计算(激活前)结果 (当前层的节点数, 样本数)
    
    Returns:
    A -- 当前层激活后的结果 (当前层的节点数, 样本数)
    cache -- 储存本次的Z
    """
  
    A = 1/(1+np.exp(-Z))
    cache = Z
	assert(A.shape == Z.shape)
    return A, cache
def relu(Z):
    """
    执行ReLU激活

    输入:
    Z -- 当前层的线性计算(激活前)结果 (当前层的节点数, 样本数)
    
    Returns:
    A -- Z -- 当前层激活后的结果 (当前层的节点数, 样本数)
    """
    
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache
	
# 前向传播计算
def linear_activation_forward(A_prev, W, b, activation):
    """
    激活线性计算结果

    输入:
    A_prev -- 上一层激活后的计算结果 (前一层的节点数, 样本数)
    W -- 参数矩阵 (当前层节点数, 前一层的节点数)
    b -- 偏置项向量	(当前层节点数, 1)
    activation -- 使用的激活函数: "sigmoid" 或 "relu"

    输出:
    A -- 当前层的激活后的计算结果 (当前层的节点数, 样本数)
    cache -- 储存计算过程中的结果，linear_cache包括A_prev/W/b，activation_cache包含Z
    """
    
    if activation == "sigmoid":
		# 线性计算
        Z, linear_cache=linear_forward(A_prev, W, b)
		# 激活计算
        A, activation_cache=sigmoid(Z)
    
    elif activation == "relu":
        # 线性计算
        Z, linear_cache=linear_forward(A_prev, W, b)
		# 激活计算
        A, activation_cache=relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
	
# 合并多层前向传播计算结果
def L_model_forward(X, parameters):
    """
    执行多层前向传播计算
    
    Arguments:
    X -- 特征矩阵 (输入层节点数, 样本数)
    parameters -- 参数字典 [来自initialize_parameters_deep或update_parameters的当前计算结果]
    
    Returns:
    AL -- 输出蹭的计算值 (输出层节点数, 样本数)
    caches -- 缓存字典: 包含每一层的linear_cache和activation_cache
    """

    caches = []
    A = X									  # 初始A相当于特征矩阵
    L = len(parameters) // 2                  # 因为每层都包括W和b两组参数，所以参数字典的长度除以2就是层数
    
    # 执行隐藏层的前向传播，并将结果加入缓存字典(隐藏层的激活函数为ReLU)
    for l in range(1, L):
        # 更新A_prev的结果，实际是上一轮的计算结果
		A_prev = A 
		# 计算本层激活后的结果
        A, cache = linear_activation_forward(A, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
		# 保存W/b/Z/A_prev
        caches.append(cache)
    
    # 执行输出层的前向传播，并将结果加入缓存字典(输出层的激活函数为Sigmoid)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    assert(len(caches) == L)
            
    return AL, caches
	
# 计算损失
def compute_cost(AL, Y):
    """
    计算损失函数的结果

    输入:
    AL -- 预测概率 [来自L_model_forward]
    Y -- 真实结果

    Returns:
    cost -- 交叉熵损失值
    """
    
    m = Y.shape[1]

    cost=-(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    
    cost = np.squeeze(cost)      # 将损失值的形式改为一个数字value (之前是[[value]])
    assert(cost.shape == ())
    
    return cost
	
# 线性梯度计算(dW/db/dA_prev)
def linear_backward(dZ, cache):
    """
    求解线性部分的梯度

    输入:
    dZ -- 当前层的线性输出的梯度 (当前层的节点数, 样本数)
    cache -- 正向传播中的缓存值， 包括当前层W/b和前一层的A(A_prev)

    Returns:
    dA_prev -- 上一层激活后结果的梯度
    dW -- 参数W的梯度
    db -- 偏置项b的梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1] # 样本数

    dW=1/m*np.dot(dZ, A_prev.T)
    db=1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev=np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# 激活函数梯度计算
def relu_backward(dA, cache):
    """
    ReLU激活函数的梯度计算

    输入:
    dA -- 当前层激活后结果的梯度
    cache -- 前向传播中当前层Z的缓存结果

    输出:
    dZ -- 当前线性结果Z的梯度
    """
    
    Z = cache
	# 梯度计算
    dZ = np.array(dA, copy=True) .
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def sigmoid_backward(dA, cache):
    """
    Sigmoid激活函数的梯度计算

    输入:
    dA -- 当前层激活后结果的梯度
    cache -- 前向传播中当前层Z的缓存结果

    输出:
    dZ -- 当前线性结果Z的梯度
    """
    
    Z = cache
    # 梯度计算
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)

    return dZ
	
# 激活函数梯度计算
def linear_activation_backward(dA, cache, activation):
    """
    线性结果到激活后结果的梯度计算
    
    输入:
    dA -- 当前层激活后结果的梯度
    cache -- 缓存值: 包含linear_cache和activation_cache
    activation -- 激活函数类型: "sigmoid" 或 "relu"
    
    Returns:
    dA_prev -- 前一层的激活后结果的梯度
    dW -- 当前层参数矩阵W的梯度
    db -- 当前层偏置项b的梯度
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        # 激活函数梯度计算
        dZ=relu_backward(dA, activation_cache)
		# 线性部分梯度计算(dW/db/dA_prev)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        # 激活函数梯度计算
        dZ=sigmoid_backward(dA, activation_cache)
		# 线性部分梯度计算(dW/db/dA_prev)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
	
# 合并反向传播过程
def L_model_backward(AL, Y, caches):
    """
    执行多层反向传播计算
    
    输入:
    AL -- 正向传播的数据结果 [来自L_model_forward]
    Y -- 标签
    caches -- 缓存字典: 包含每一层的linear_cache和activation_cache [来自L_model_forward]
    
    输出:
    grads -- 梯度字典，包括每一层的A/W/b的梯度
    """
    grads = {}
    L = len(caches) # 层数
    m = AL.shape[1] # 样本数
    Y = Y.reshape(AL.shape)
    
    # 开始反向传播
    # 输出层L的激活值AL的梯度
    grads['dA'+str(L)]=- (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # 输出层参数W和b，以及前一层激活值AL-1的梯度
    current_cache=caches[-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)]=linear_activation_backward(grads['dA'+str(L)], current_cache, "sigmoid")
    
    # 从L-2值输入层的遍历循环
    for l in reversed(range(L-1)):
        # 计算该层W和b的梯度，以及前一层A的梯度
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
	
# 更新参数
def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降方法更新参数
    
    输入:
    parameters -- 参数字典 [来自initialize_parameters_deep或update_parameters的前一轮计算结果]
    grads -- 梯度字典 [来自L_model_backward]
    
    输出:
    parameters -- 更新后的参数字典
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # 更新每一层的参数
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters
	
# 整合后的深度神经网络模型
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): #lr was 0.009
    """
    执行多层神经网络模型，输出优化后的参数
    
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    layers_dims -- 网络结构，包括每层的节点数
    learning_rate -- 学习速率
    num_iterations -- 迭代轮次
    print_cost -- 显示损失函数变化结果
    
    输出:
    parameters -- 最优化后的模型惨呼
    """

    np.random.seed(1)
    costs = []                         
    
    # 参数初始化
    parameters = initialize_parameters_deep(layers_dims)
    
    # 迭代更新参数
    for i in range(0, num_iterations):

        # 前向传播计算预测结果
        AL, caches=L_model_forward(X, parameters)
        
        # 计算损失值
        cost=compute_cost(AL, Y)
    
        # 反向传播计算梯度
        grads=L_model_backward(AL, Y, caches)
 
        # 更新参数
        parameters=update_parameters(parameters, grads, learning_rate)
                
        # 显示损失函数变化过程
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # 损失值的折线图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
	
# 预测
def predict(X, y, parameters, print_accuracy=True):
    """
    利用深度神经网络模型进行预测
    
    输入:
    X -- 特征矩阵
    parameters -- 参数
    
    Returns:
    p -- 模型预测结果
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # 前向传播计算预测概率
    probas, _ = L_model_forward(X, parameters)

    
    # 利用预测概率预测标签
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    if print_accuracy: print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return probas, p