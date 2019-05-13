import numpy as np
# layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# pooling layers
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# model
from keras.models import Sequential
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import pydot
from IPython.display import SVG
from kt_utils import *

# 设计模型结构
def Model(input_shape):
    """
    设计模型结构
    
    输入:
    input_shape -- 数据形式

    输出:
    model -- Keras模型
    """
    
    # 模型
    model = Sequential()

    # 添加第一个卷积: 卷积->BN->激活->池化
    model.add(Conv2D(filters=8, kernel_size=(3,3), strides = (1, 1), padding='same', input_shape=input_shape, name = 'conv1'))
    model.add(BatchNormalization(axis = 3, name = 'bn1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), name='max_pool1'))
    
    # 添加第一个卷积: 卷积->BN->激活->池化
    model.add(Conv2D(filters=16, kernel_size=(3,3), strides = (1, 1), padding='same' ,name = 'conv2'))
    model.add(BatchNormalization(axis = 3, name = 'bn2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), name='max_pool2'))

    # 添加第一个卷积: 卷积->BN->激活->池化
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides = (1, 1), padding='same' ,name = 'conv3'))
    model.add(BatchNormalization(axis = 3, name = 'bn3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), name='max_pool3'))
    
    # 添加三个全连接曾
    model.add(Flatten()) # 先将卷积层拉平化
    model.add(Dense(128, activation='relu', name='fc4'))
    model.add(Dense(64, activation='relu', name='fc5'))
    model.add(Dense(32, activation='relu', name='fc6'))
	
	# 输出
    model.add(Dense(1, activation='sigmoid', name='out'))
    
    return model

# 显示模型结构
print(happyModel.summary())
plot_model(happyModel, to_file='HappyModel.png')

# 编译模型
happyModel.compile(loss='binary_crossentropy',  # 损失函数
        optimizer="Adam",					    # 优化器
        metrics=['accuracy'])				    # 评估指标

# 训练模型
happyModel.fit(x=X_train, y=Y_train,			# 数据集		
              batch_size=16, epochs=10,			# batch size和epoch
              validation_data=(X_test, Y_test), # 验证集
              shuffle=True)						# 每个epoch重新shuffle训练集

# 评估结果
loss, accuracy = happyModel.evaluate(X_test,Y_test)

# 预测样本
happyModel.predict(x)