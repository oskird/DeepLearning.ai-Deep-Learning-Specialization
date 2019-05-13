# 恒等残差块
def identity_block(X, f, filters, stage, block):
    """
    恒等残差块
    
    输入:
    X -- 输入矩阵
    f -- 过滤器大小
    filters -- 每个卷积层的过滤器个数(信道数)
    stage -- 用于识别网络层所在的阶段
    block -- 用于识别网络层所在的阶段
    
    输出:
    X -- 输出矩阵
    """
    
    # 定义命名方式
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 获取各层过滤器大小
    F1, F2, F3 = filters
    
    # 设置shortcut路径
    X_shortcut = X
    
    # 主路径第一个成分
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # 主路径第二个成分
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # 主路径第三个成分
	# 主路径卷积和BN部分
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    # 将shortcut部分加入主路径
    X = Add()([X,X_shortcut])
	# 第三个成分的激活
    X = Activation("relu")(X)
    
    return X
	
# 卷积残差块
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    卷积残差块
    
    输入:
    X -- 输入矩阵
    f -- 过滤器大小
    filters -- 每个卷积层的过滤器个数(信道数)
    stage -- 用于识别网络层所在的阶段
    block -- 用于识别网络层所在的阶段
    s -- 步长
    
    输出:
    X -- 输出矩阵
    """
    
    # 定义命名方式
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 获取各层过滤器大小
    F1, F2, F3 = filters
    
    # 设置shortcut路径
    X_shortcut = X

    # 主路径第一个成分
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # 主路径第二个成分
    X = Conv2D(F2, (f, f), strides = (1,1),padding = "same", name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # 主路径第三个成分
	# 主路径卷积和BN部分
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    # 对shortcut添加卷积计算
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    # 将shortcut部分加入主路径
    X = Add()([X,X_shortcut])
	# 第三个成分的激活
    X = Activation('relu')(X)
    
    return X

# ResNet50
def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    输入:
    input_shape -- 输入图像矩阵的形状
    classes -- 预测类别数

    Returns:
    model -- Keras模型
    """
    
    # 定义输入形式
    X_input = Input(input_shape)

	# 填充
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1：卷积层->BN->激活->池化
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2：一个卷积残差块+两个恒等残差块
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3：一个卷积残差块+三个恒等残差块
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4：一个卷积残差块+五个恒等残差块
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5：一个卷积残差块+两个恒等残差块
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='d')

    # 平均池化
    X = AveragePooling2D(pool_size=(2,2),name = "avg_pool")(X)

    # 添加全连接层
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

# 模型使用
model = ResNet50(input_shape = (64, 64, 3), classes = 6) 							   	# 设置模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 	# 编译模型
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)							   	# 训练模型
loss, accuracy = model.evaluate(X_test, Y_test) 									   	# 效果评估