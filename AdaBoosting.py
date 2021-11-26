#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Description: 复现西瓜书AdaBoost算法，对应p174图8.3, 基学习器为单层的决策树
Date: 2021/11/25 09:58:21
Author: Yanzhan Chen
version: 1.0
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset():
    """
    description:生成西瓜3.0a的数据集
    ---------
    param: none
    -------
    Returns: 训练集的X和y,均为array类型的数据
    -------
    """
    data = pd.read_csv('./西瓜数据集.txt',sep=',')
    X = data.iloc[:,:-1].values   #X为(size,2)的array,表示特征
    y = data.iloc[:,-1].values.reshape(-1,1)    #y为(size,1)的array,表示标签
    print(X.shape)
    print(y.shape)
    return X, y


def SingleTree(X, dimen, threshold, inequal='lt'):
    """
    description: 建立一个单层决策树的模型，但该模型没有进行训练，需要后续的函数进行筛选，选择合适的切分点
    ---------
    param:
    X ：输入的特征数据，(size,2)-->array
    dimen: 特征维度，(0 or 1)
    threshold: 切分点，[X.[:,dime].min(),X.[:,dime].max()]
    inequal: str ('lt' or 'gt') 当特征大于或小于threshold时判别为-1，经此两种情况
    -------
    Returns: 决策树对训练集的判别结果 (size,1)-->array
    -------
    """
    m, n = X.shape
    result = np.ones((m,1))
    if inequal == 'lt':
        result[X[:,dimen]<threshold] = -1
    else:
        result[X[:,dimen]>threshold] = -1
    
    return result

def TrainModel(X, y, D, verbose=True):
    """
    description:训练一个误差最低的单层决策树二分类器
    ---------
    param:
    X: 输入训练数据的特征 (size,2)
    y: 输入训练数据的标签 (size,1)
    D: 每个样本的频率，随着集成的次数增多，每个样本不是均匀分布，D是一个(size,1)的array
    verbose: 控制是否打印切分节点的具体信息
    -------
    Returns:
    -------
    """
    m, n = X.shape # m:样本的个数，n:样本的特征数
    splitNum = 10 #每个特征为数值类型，设置均匀切分为10份
    MinError = float('inf') #记录最优模型的分类误差
    BestTree = {}
    #遍历每个特征的切分点，筛选最佳的切分点和inequal，相当于训练单层决策树模型
    for atrr in range(n):
        atrrMin = X[:,atrr].min()
        atrrMax = X[:,atrr].max()
        step = (atrrMax - atrrMin)/splitNum
        for i in range(-1,splitNum+2):
            threshold = atrrMin + i*step
            for category in ['lt','gt']:
                ErrorArray = np.zeros((m,1)) #记录预测错误的样本，正确为0，错误为1，先初始化为0
                predict = SingleTree(X, atrr, threshold, category)
                ErrorArray[predict!=y] = 1
                TotalError = np.dot(ErrorArray.T,D)[0,0]
                if (verbose):
                    print('第%d个特征，切分点为%.4f,误差为%.4f' %(atrr,threshold,TotalError))
                if TotalError < MinError:
                    MinError = TotalError
                    BestTree['dimen'] = atrr
                    BestTree['threshold'] = threshold
                    BestTree['inequal'] = category
                    PredictLabel = predict
    return MinError, BestTree, PredictLabel

def TrainAdaboost(X, y, Epochs = 40):
    """
    description: AdaBoost算法核心过程，对应西瓜书p174图8.3
    ---------
    param:
    X：训练数据的特征
    y: 训练数据的标签
    Epochs: 迭代次数，对应学习器的个数
    -------
    Returns: 保存每个基学习器信息的列表
    -------
    """
    LearnerHistory = [] #保存基学习器的列表
    m,n = X.shape
    D = np.ones((m,1))*(1/m)  #初始化样本的概率分布
    AllPredict = np.zeros((m,1)) #记录所有已训练模型的集成预测结果，初始化所有元素为0

    #开始集成学习的训练过程
    for epoch in range(Epochs):
        #训练当前迭代下的基学习器
        MinError, BestTree, PredictLabel = TrainModel(X,y,D,verbose=False)
        #当期望误差大于0.5时候，退出循环
        if MinError > 0.5:
            break
        
        #计算当前迭代模型的权重
        alpha = 0.5*np.log((1-MinError)/max(MinError,1e-16)) #防止MinError为0导致报错

        BestTree['alpha'] = alpha
        print("第%d次迭代决策树模型第%d个特征, 切分点为%.3f, ineqal: %s,总误差为%.3f, 权重为：%.3f" % (epoch+1, BestTree['dimen'], BestTree['threshold'], BestTree['inequal'], MinError, BestTree['alpha']))
        LearnerHistory.append(BestTree)
        #更新样本的概率分布D
        expon = -alpha*(y*PredictLabel)
        D = D * np.exp(expon)
        D = D/D.sum()  # p174图8.3第7行  根据样本权重公式，更新样本权重

        #计算前epoch+1个基学习器的集成算法的误差
        AllPredict += alpha*PredictLabel
        ErrorCount = np.ones((m,1)) #记录AllPredict与y中不同的预测结果
        AllPre = np.sign(AllPredict)
        print("前{}个弱分类器得到的结果:{} ".format(epoch+1, AllPre.T))
        ErrorCount[AllPre == y] = 0
        ErrorRate = ErrorCount.sum()/m
        print('分类错误率: %.3f' %ErrorRate)
        #如果错误率为0了，表示训练效果已经可以了，可不需再集成了
        if ErrorRate == 0:
            break
    
    return LearnerHistory

def visualization(X, y, EnsembleModel):
    """
    description: 可视化训练模型的结果，包含每个分类器的划分点
    ---------
    param:
    X: 输入特征
    y: 输入标签
    EnsembleModel: 集成学习训练的模型，存储在一个元素为字典的列表中
    -------
    Returns: None
    -------
    """
    m,n = X.shape
    xlimit = [X[:,0].min(),X[:,0].max()]
    ylimit = [X[:,1].min(),X[:,1].max()]
    GoodMelon = []
    BadMelon = []
    for i in range(m):
        if y[i,-1] > 0:
            GoodMelon.append(X[i,:].tolist())
        else:
            BadMelon.append(X[i,:].tolist())
    GoodMelon = np.array(GoodMelon)
    BadMelon = np.array(BadMelon)

    plt.rc('font',family='Times New Roman')
    plt.scatter(GoodMelon[:,0],GoodMelon[:,1],s=30,c='red',marker='o',alpha=0.5,label='Good')
    plt.scatter(BadMelon[:,0],BadMelon[:,1],s=30,c='blue',marker='x',alpha=0.5,label='Bad')

    for baseleaner in EnsembleModel:
        print(baseleaner)
        if baseleaner['dimen'] == 0:
            plt.plot([baseleaner['threshold'],baseleaner['threshold']],ylimit,linestyle=':')
        else:
            plt.plot(xlimit,[baseleaner['threshold'],baseleaner['threshold']],linestyle=':')
    
    plt.legend()
    plt.xlabel('density')
    plt.ylabel('Sugar content')
    plt.show()
    return None

if __name__ == '__main__':
    X,y = load_dataset()   #创建数据集
    m,n = X.shape
    # print(SingleTree(X,0,0.3,'lt'))
    # D = np.ones((m,1))*(1/m)
    # MinError, BestTree, PredictLabel = TrainModel(X,y,D)
    # print(BestTree)
    # print(MinError)

    model = TrainAdaboost(X, y, 40) #训练AdaBoost模型
    visualization(X,y,model)   #可视化