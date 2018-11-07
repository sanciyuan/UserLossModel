#!/usr/bin/env python
# coding: utf-8

# 读取用户流失测试数据

#载入pandas包来读取csv格式的数据集
import pandas as pd

#把 csv格式的数据集导入到DataFrame对象中
df = pd.read_csv('lossertest.csv',  header = 0)
# print(df.head())


# 首先要对元数据进行数据清理，需要把所有数据转化为数值型数据

#把totalPaiedAmount列也就是用户付款金额的缺失值替换为0
df['totalPaiedAmount'] = df['totalPaiedAmount'].fillna(0)
df['totalBuyCount'] = df['totalBuyCount'].fillna(0)
# print(df.head())


#利用pandas中的to_datetime函数把字符串的日期变为时间序列
df['registrationTime'] = pd.to_datetime(df['registrationTime'], format='%Y-%m-%d %H:%M:%S')
# print(df['registrationTime'])



#同理转化为实践序列
df['lastLoginTime'] = pd.to_datetime(df['lastLoginTime'], format='%Y-%m-%d %H:%M:%S') 
# print(df['lastLoginTime'])



import datetime
#获取当前时间
now_time = datetime.datetime.now()
# print(now_time)


#把数据序列转化为距今的时间间隔
df['registrationTime'] = now_time-df['registrationTime']
# print(df['registrationTime'])


df['lastLoginTime'] = now_time-df['lastLoginTime']
# print(df['lastLoginTime'])


#把最近登录时间列的空值替换为同索引行注册时间列的值
df.loc[df['lastLoginTime'].isnull(),'lastLoginTime']=df[df['lastLoginTime'].isnull()]['registrationTime']
# print(df['registrationTime'])



#因为数据量有点大，取前1w行数据测试下
df = df.iloc[0:1000]

#把时间间隔转化为数值型的天数
j = 0
for i in df['registrationTime']:
    df = df.replace(df['registrationTime'][j],i.days)
    j += 1

j = 0
for i in df['lastLoginTime']:
    df = df.replace(df['lastLoginTime'][j],i.days)
    j += 1
# print(df['lastLoginTime'])


#对数据集进检查，看看是否还有缺失值
df[df.isnull().values==True]


#把第一列无用的用户ID列删除
df = df.iloc[:,1:]

# print(df.head())
# print(df.info())


#把输入输出项确定下
#前6列是输入的指标，最后一列流失标记是输出项
y = df.iloc[:,-1]
x = df.iloc[:,:-1]
# print(x.shape)
# print(y.shape)


# 区分训练与测试数据集

#sklearn把数据集拆分成训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 123)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# 尺度标准化

#使用sklearn把数据集进行尺度标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# print(x_test)


# 训练ANN

#使用keras包搭建人工神经网络
# 使用tensorflow中集成的Keras包，替代Keras包
import tensorflow.contrib.keras as kr
from tensorflow.contrib.keras import *

# import keras
# #序贯（Sequential）模型包
# from keras.models import Sequential
# #神经网络层
# from keras.layers import Dense
# #优化器
# from keras.optimizers import SGD


#创建一个空的神经网络模型
classifier = kr.models.Sequential()


#创建输入层
classifier.add(kr.layers.Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
#创建输出层
classifier.add(kr.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(loss='binary_crossentropy',
              optimizer=kr.optimizers.SGD(),
              metrics=['accuracy'])

history = classifier.fit(x_train, y_train,
                    batch_size=10,
                    epochs=100,
                    validation_data=(x_test, y_test))


# 评估模型

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)
print(y_pred.shape) # (330, 1)

y_pred = y_pred.flatten().astype(int)
print(y_pred)
# print('准确率为：',sum(y_pred.flatten().astype(int) == y_test) / len(y_test))

from sklearn.metrics import accuracy_score
print('准确率为：', accuracy_score(y_test, y_pred ))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred )
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))




