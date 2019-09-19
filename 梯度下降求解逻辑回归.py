"""
逻辑回归 python实践
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pdData = pd.read_csv('/home/tarena/机器学习算法实践/test_datas/LogiReg_data.txt',header=None,
                     names=['Exam 1','Exam 2','Admitted'])
# positive = pdData[pdData['Admitted']==1]
# negative = pdData[pdData['Admitted']==0]

# fig,ax = plt.subplots(figsize=(10,5))
# ax.scatter(positive['Exam 1'],positive['Exam 2'], s=30,c='b',marker='o',label='Admitted')
# ax.scatter(negative['Exam 1'],negative['Exam 2'], s=30,c='r',marker='x',label='Not Admiited')
# ax.legend()
# ax.set_xlabel('Exam 1 score')
# ax.set_ylabel('Exam 2 score')
#The logistic regression

#目标 建立分类器（求解出三个参数）
#定阈值， 0.5

# 要完成的模块
#sigmoid：映射到概率的函数
#model：返回预测结果值
#cost：根据参数计算损失
#gradient：计算没一个参数的梯度方向
#descent：进行参数更新
#accuracy：计算精度


#1111-————sigmoid函数  g(z) = 1/1+e的-z次方
def sigmiod(z):
    """
    函数
    取值范围 正负无穷
    值域：【0，1】
    :param z:
    :return:
    """
    return 1/(1+np.exp(-z))

def model(x,theta):
    #预测函数 将输转化成sigmoid 函数
    return sigmiod(np.dot(x,theta.T))

pdData.insert(0,'Ones',1)
orig_data = pdData.as_matrix()
cols = orig_data.shape[1]
X=orig_data[:,0:cols-1]
y=orig_data[:,cols-1:cols]

theta = np.zeros([1,3])  #制定theta 站位 aX + bX2
print(theta)
#损失函数---->对数似然
def cost(X,y,theta):
    left = np.multiply(-y,np.log(model(X,theta)))
    right = np.multiply(1-y,np.log(1-model(X,theta)))
    return np.sum(left-right)/(len(X))
print('损失函数',cost(X,y,theta))
# 计算梯度
def gradient(X,y,theta):
    '''计算每个梯度的更新方向'''
    grad = np.zeros(theta.shape)
    error = (model(X,theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error,X[:,j])
        grad[0,j] = np.sum(term) / len(X)
    return grad
# 比较三种梯度下降  {批量梯度下降， 随机梯度下降，小批量梯度下降}
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type,value,threshold):
    print(np.linalg.norm(value))
    #设定三种不同的停止侧率
    if type == STOP_ITER:   return value >threshold
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold

import numpy.random
#洗牌
def suffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:,0:cols-1]
    y = data[:,cols-1:]
    return X,y
import time                                # thresh 策略对应的阈值e:
def descent(data,theta,batchSize,stopType,thresh,alpha):
    #梯度下降求解
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    X,y = suffleData(data)
    grad = np.zeros(theta.shape)    # 计算的梯度
    costs = [cost(X,y,theta)]       #损失值

    while True:
        grad = gradient(X[k:k+batchSize],y[k:k+batchSize],theta)
        k += batchSize
        if k >= n:
            k = 0
            X,y = suffleData(data)
        theta = theta - alpha*grad
        print(theta)
        costs.append(cost(X,y,theta))
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType,value,thresh):
            break

    return theta, i-1,costs,grad,time.time()-init_time
# 选择梯度下降方法给予所有样本

### STOP_ITER
n = 100
# result_theta,i,list,grad,time = descent(orig_data,theta,n,STOP_ITER,thresh=5000,alpha=0.000001)  #整体梯度下降 5000次循环
# print('结果：',result_theta,'时间',time,'次数',i,'损失值：',list[-1])
# 结果： [[-0.00064506  0.00073214  0.01059727]] 时间 1.1436514854431152 次数 5000 损失值： 0.6332713628744226

###STOP_COST
# result_theta,i,list,grad,time = descent(orig_data,theta,n,STOP_COST,thresh=0.000001,alpha=0.001) #thresh 最后差值小鱼0.000001停止
# print('损失值：',list[-1],'次数：',i)
#损失值： 0.1271508375783415 次数： 101424
###STOP_ITER
# result_theta,i,list,grad,time = descent(orig_data,theta,n,STOP_ITER,thresh=15000,alpha=0.000001) #thresh 最后差值小鱼0.000001停止
# print('结果：',result_theta,time,i,'损失值：',list[-1])

#发现 结果不太理想  进行数据标准化
from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:,1:3] == pp.scale(orig_data[:,1:3])

# result_theta,i,list,grad,time = descent(scaled_data,theta,n,STOP_ITER,thresh=10000,alpha=0.001)
# print('结果：',result_theta,time,i,'损失值：',list[-1])
#结果： [[-1.1403195  -0.04472882  0.07824091]] 2.1802544593811035 10000 损失值： 0.442228277170268
# result_theta,i,list,grad,time = descent(scaled_data,theta,16,STOP_GRAD,thresh=0.001,alpha=0.001)
# print("STOP:GRAD:!!"'结果：',result_theta,time,i,'损失值：',list[-1])
 #batch 比较适合

# 设定阈值
def predict(X,theta):
    return [1 if x>=0.5 else 0 for x in model(X,theta)]


result_theta,i,list,grad,time = descent(scaled_data,theta,n,STOP_ITER,thresh=15000,alpha=0.001)
scaled_X = scaled_data[:,:3]
y = scaled_data[:,3]
predictions = predict(scaled_X,result_theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(predictions,y)]
accuracy = sum(map(int,correct)) % len(correct)
print('accuracy = {0}%'.format(accuracy))
