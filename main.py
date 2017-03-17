#!usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
from numpy import *
import pandas as pd
from sklearn.metrics import roc_curve #导入ROC曲线函数
import matplotlib.pyplot as plt


# calculate the sigmoid function
def sigmoid(z):
    return 1.0 / (1 + exp(-z))

# m denotes the number of examples here, not the number of features
def gradientDescent(train_x, train_y, theta, alpha, m, num_n, num_iterations):
    x_trans =train_x.transpose()
    for i in range(0, num_iterations):
        hypothesis = sigmoid(dot(train_x, theta))
        error = hypothesis - train_y
        gradient = dot(x_trans, error) / m
        # update
        theta = theta - alpha * gradient

    return theta


def cleanData(original_data_file, output_file):
    original_df = pd.read_excel(original_data_file, header=0)
    original_num_samples, original_num_features = shape(original_df)
    original_df.columns = range(original_num_features)
    for i in range(original_num_features):
        for j in range(original_num_samples):
            original_df[i][j] = (original_df[i][j] - original_df[i].mean()) / original_df[i].std()
    original_df.to_excel(output_file)


def getFinalData(final_data_file, p):
    df = pd.read_excel(final_data_file, header=0)
    df = df.as_matrix()
    num_m, num_n = shape(df)
    x_0 = ones(num_m)  # 初始化 x_0
    df = insert(df, 0, x_0, axis=1)  # 插入x0 = 1
    # 划分训练集与测试集
    train_x = df[0:int(len(df)*p), range(num_n)]
    train_y = df[0:int(len(df)*p), [num_n]] #注意啦，这里一定要两个中括号，返回值类型为 'DataFrame'，不然的话，返回的 train_y 就是 'Series'
    test_x = df[int(len(df)*p):, range(num_n)]
    test_y = df[int(len(df)*p):, [num_n]]
    num_samples, num_features = shape(train_x)
    alpha = 0.01
    num_iterations = 1000
    theta = ones((num_features, 1))
    theta = gradientDescent(train_x, train_y, theta, alpha, num_samples, num_features, num_iterations)

    return theta, test_x, test_y


def testLogRegres(theta, test_x, test_y):
    num_samples, num_features = shape(test_x)
    theta_trans = theta.transpose()
    scores = []
    for i in range(num_samples):
        z = dot(theta_trans, test_x[i])
        predict = sigmoid(z)
        if predict >= 0.5:
            predict = 1
        else:
            predict = 0
        scores.append(predict)
    print(scores)

    #用 ROC 曲线评估模型
    fpr, tpr, thresholds = roc_curve(test_y, scores, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')  # 作出ROC曲线
    plt.xlabel('False Positive Rate')  # 坐标轴标签
    plt.ylabel('True Positive Rate')  # 坐标轴标签
    plt.ylim(0, 1.05)  # 边界范围
    plt.xlim(0, 1.05)  # 边界范围
    plt.legend(loc=4)  # 图例
    plt.show()  # 显示作图结果a

    cm = confusion_matrix(test_y, scores)  # 混淆矩阵
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()

if __name__ == '__main__':
    original_data_file = '../data/original-data.xls'
    final_data_file = '../data/finaldata.xls'
    cleanData(original_data_file, final_data_file)
    p = 0.8  #训练集所占的比例
    theta, test_x, test_y = getFinalData(final_data_file, p)
    testLogRegres(theta, test_x, test_y)
