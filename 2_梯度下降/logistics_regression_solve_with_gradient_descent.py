# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:15:17 2018

@author: a
"""

#%% ============================Logistic Regression=======================
"""
我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。
假设你是一个大学系的管理员，你想根据两次考试的结果来决定每个申请人的录取机会。
你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。
对于每一个培训例子，你有两个考试的申请人的分数和录取决定。
为了做到这一点，我们将建立一个分类模型，根据考试成绩估计入学概率。
"""

"""预读笔记：重要知识点!
-logistic 回归的前向过程：sigmoid
-梯度下降法

#######
-plot_decision_boundary：
        绘制分类数据的logistics回归分界线及区域,可做模板！！！！！
        
-sklearn.linear_model.LogisticRegressionCV().fit(X.T, Y.T)
        事实上可以利用sklearn中的LogisticRegressionCV函数训练模型，注意输入输出即可，相应的绘制边界图像也有小有修改
"""



#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline


#%% 
import os 
path = 'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam1','Exam2','Admitted'])
pdData.head()

pdData.shape#(100,3)

#%% 数据可视化：线性可分，考虑使用logistic 回归
positive = pdData[pdData['Admitted']==1]#returns the subset of rows such Admitted = 1, i.e. the set of *positive* examples
negative = pdData[pdData['Admitted']==0]#returns the subset of rows such Admitted = 0, i.e. the set of *positive* examples

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['Exam1'],positive['Exam2'],s=30,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=30,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')

#%%
"""#关于Logistic回归，需要完成以下模块（函数）
-sigmoid : 映射到概率的函数
-model : 返回预测结果值——————f(X,θ),由于自变量X是二维数据参数，则θ=(θ0,θ1,θ2)
-cost : 根据参数计算损失
-gradient : 计算每个参数的梯度方向
-descent : 进行参数更新
-accuracy: 计算精度
"""

#%%sigmoid
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
'''
nums=np.arange(-10,10,step=1)
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(nums,sigmoid(nums),'b')
'''

#%%model
def logistics_model(X, theta):
    return sigmoid(np.dot(X,theta.T))

pdData.insert(0,'Ones',1)
orig_data = pdData.as_matrix()
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]#注意，不能直接用-1或者cols-1，这样单独取列的话返回的就是一组行向量，应该用切片的方法

theta = np.zeros([1,3])#1*3行向量

#%%loss function
def cost(X,y,theta):
    loss_1 = - np.multiply(y,np.log(logistics_model(X,theta)))
    loss_2 = - np.multiply(1-y,np.log(1-logistics_model(X,theta)))
    return np.sum(loss_1+loss_2) / (X.shape[0])
#cost(X,y,theta)


#%%gradient and gradient descent
def gradient(X,y,theta):
    grad = np.zeros(theta.shape)
    error = (logistics_model(X,theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error,X[:,j])
        grad[0,j] = np.sum(term) / len(X)
    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    #设置GD停止策略
    if   type == STOP_ITER: return value > threshold
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold
    
import numpy.random
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

import time
def gradient_descent(data, theta, batchsize, stopType, thresh, alpha):
    init_time = time.time()
    i = 0#epoch number
    k = 0
    X, y =shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]

    while True:
        grad = gradient(X[k:k+batchsize],y[k:k+batchsize],theta)#取一个batch计算一次梯度
        k += batchsize
        if k>= n:
            k = 0
            X, y = shuffleData(data)#重新洗牌
            
        theta = theta -alpha*grad
        costs.append(cost(X, y, theta))
        i += 1
        
        if   stopType == STOP_ITER: value = i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        if stopCriterion(stopType, value, thresh):break
    
    return theta, i-1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = gradient_descent(data, theta, batchSize, stopType, thresh, alpha)
    
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += "data - learning rate : {} - ".format(alpha)
    
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1: strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + "descent - Stop: "
    
    if stopType == STOP_ITER: strStop = " {} iterations ".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
            
    print("*** {} \nTheta: {} - Iter: {} -Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1], dur))        
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta
            
#%%对比不同的停止策略
#设定停止次数
n = 100#总共就100个样本
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000,alpha=0.000001)
      
#根据损失值停止
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)            

#根据梯度变化停止
runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)            


#%%对比不同的梯度下降法
#Stochastic descent
runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
   #由图结果决定将学习率调低
runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)


#Mini-batch descent
runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
   #浮动仍然很大
   
#%%对数据进行标准化后做实验
from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])

runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)
"""
原始数据，只能达到达到0.61，而在这里我们得到了0.38！ 
所以对数据做预处理是非常重要的
"""
runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)
      #更多的迭代次数会使得损失下降的更多

theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
      #随机梯度下降更快，但是我们需要迭代的次数也需要更多，所以还是用batch的比较合适！！！

runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)


#%% 精度
def predict(X,theta):
    
    #return [1 if x>= 0.5 else 0 for x in logistics_model(X,theta)]
    return (logistics_model(X,theta) >= 0.5)

scaled_X = scaled_data[:,:3]
y = scaled_data[:,3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions, y)]
accuracy = sum(correct) % len(correct)
print ('accuracy = {0}%'.format(accuracy))


#%%plot_decision_boundary
#"""待完成
def plot_decision_boundary(model, X, y):
    '''
    X:    变量
    y:    分类
    model:训练得到的预测模型
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 1].min() - 1, X[:, 2].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 2].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))#返回值是np.ndarray
    '''
    当做惯例就好，按0.01的间隔绘制网格图，得到匹配的每个格点，之后就对每个点做预测
    '''
    
    
    # Predict the function value for the whole grid
    b = np.ones((xx.shape[0]*xx.shape[1],1))
    Z = model(np.column_stack((b,np.c_[xx.ravel(), yy.ravel()])),theta)
    Z = np.array(Z,dtype = int)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha = 0.9)
    '''
    绘制等高热度图，xx、yy对应横纵坐标，Z为相应的高度（也即分类值），alpha改变颜色透明度
    '''
    
    line = plt.contour(xx, yy, Z, colors = 'black', alpha = 0.5, linewidths = 0.5)
    '''
    绘制等高线，在这里相当于分界线。注意画分界线之前必须现有等高的区域
    '''
    
    plt.scatter(X[:, 1], X[:, 2], c=y, edgecolors='black',cmap=plt.cm.Spectral)
    plt.title('Decision Boundary for Logistic Regression of scaled data')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    
plot_decision_boundary(predict, scaled_X, y)
#"""
