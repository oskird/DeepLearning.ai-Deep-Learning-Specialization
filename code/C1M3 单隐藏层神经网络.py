import numpy as np

def layer_sizes(X, Y, hidden=4):
    """
	设置网络层数
	
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    
    输出:
    n_x -- 输入层节点数(特征数)
    n_h -- 隐藏层节点数
    n_y -- 输出层节点数(1)
    """
    n_x = X.shape[0]
    n_h = hidden
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y, seed=2):
    """
	随机初始化权重
	
    输入:
    n_x -- 输入层节点数(特征数)
    n_h -- 隐藏层节点数
    n_y -- 输出层节点数(1)
    
    输出:
    参数字典:
		W1 -- 第一层权重矩阵 (隐藏层节点数, 输入层节点数)
		b1 -- 第一层偏置项 (隐藏层节点数, 1)
		W2 -- 第二层权重矩阵 (1, 隐藏层节点数)
		b2 -- 第二层偏置项  (1, 1)
    """    
    np.random.seed(seed) 
    # 随机初始化
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
	
def forward_propagation(X, parameters):
    """
	前向传播计算输出值
	
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    parameters -- 参数字典，包含W1/b1/W2/b2
    
    输出:
    A2 -- 输出层计算结果 (1, 样本数)
    cache -- 储存Z1/A1/Z2/A2的计算结果
    """
    # 获取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # 前向传播计算输出值A2
    Z1 = np.dot(W1, X)+b1
    A1 = (np.exp(Z1)-np.exp(-Z1))/(np.exp(Z1)+np.exp(-Z1))
    Z2 = np.dot(W2, A1)+b2
    A2 = 1/(1+np.exp(-Z2))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache
	
def compute_cost(A2, Y, parameters):
    """
    计算交叉熵损失值
    
    输入:
    A2 -- 输出层计算结果 (1, 样本数)
    Y -- 真实值标签 (1, 样本数)
    parameters -- 参数字典，包含W1/b1/W2/b2
    
    输出:
    cost -- 交叉熵损失函数的计算值
    """
    m = Y.shape[1]  # 样本数个数

    # 计算损失函数的结果
	cost=-(1/m)*np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))
    cost = np.squeeze(cost)
    
    return cost
	
def backward_propagation(parameters, cache, X, Y):
    """
    反向传播计算梯度
    
    Arguments:
    parameters -- 参数字典，包含W1/b1/W2/b2
    cache -- 储存Z1/A1/Z2/A2的计算结果
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    
    Returns:
    grads -- 梯度字典，包含dW1/db1/dW2/db2
    """
    m = X.shape[1]
    
    # 提取参数
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
        
    # 提取A1和A2
    A1 = cache['A1']
    A2 = cache['A2']
    
    # 反向传播计算梯度
    dZ2 = A2-Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    利用梯度下降方法更新参数
    
    输入:
    parameters -- 参数字典，包含W1/b1/W2/b2
    grads -- 梯度字典，包含dW1/db1/dW2/db2
    
    输出:
    parameters -- 更新后的参数字典，包含更新后的W1/b1/W2/b2
    """
    # 提取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # 提取梯度
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # 更新参数
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
	整合浅层神经网络算法
	
    输入:
    X -- 特征矩阵 (特征数, 样本数)
    Y -- 标签 (1, 样本数)
    n_h -- 隐藏层节点数
    num_iterations -- 迭代轮次
    print_cost -- 是否显示损失函数的变化过程
    
    Returns:
    parameters -- 参数字典，包含求解后的最优参数W1/b1/W2/b2
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    for i in range(0, num_iterations):
         
        # 前向传播计算输出值
        A2, cache = forward_propagation(X, parameters)
        
        # 计算损失值
        cost = compute_cost(A2, Y, parameters)
 
        # 反向传播计算梯度
        grads = backward_propagation(parameters, cache, X, Y)
 
        # 使用梯度下降法更新参数
        parameters = update_parameters(parameters, grads)
        
        # 显示每1000轮的迭代结果
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """
    利用学习到最优参数预测
    
    输入:
    parameters -- 参数字典，包含求解后的最优参数W1/b1/W2/b2
    X -- 特征矩阵 (特征数, 样本数)
    
    输出：
    predictions -- 预测结果，特征矩阵 (1, 样本数)，每个值为0或1
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    predictions, _=forward_propagation(X, parameters)
    
    return np.round(predictions).astype(int)