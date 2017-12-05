# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:40:03 2017

@author: HYX
"""

# 构建一个简单的神经网络
# 主要需要使用向量话来完成
# 主要元素A ，Ｗ，Z, sigmod
# 网络3*4*1的网络

# 核心公式：Z = W * A
# A'2 = sigmod(Z)

import numpy as np
from numpy import random

import math

#因此，可以发现，针对层进行编程即可
#如果可以达到构建栈的方式，那就极好了

#这样可以通过不断输入inpDim和outDim构建
def createLayer(A, inpDim, outDim):
    W = random.rand(outDim,inpDim)
    B = random.rand(outDim,1)
    Z = W.dot(A) + B
    A = 1/(1+np.exp(-Z))
    return A

def main():
    print ("hello, ")
    #前向传播核心
    A0 = random.rand(3,1)
    W1 = random.rand(4,3)
    B1 = random.rand(4,1)
    Z1 = W1.dot(A0) + B1
    A1 = 1/(1+np.exp(-Z1))

    W2 = random.rand(1,4)
    Z2 = W2.dot(A1)
    A2 = 1/(1+np.exp(-Z2))

#反向传播
'''
    dZ2 = A2 - y
    dW2 = dZ.dot(A1.T)
    dB2 = dZ
    dZ1 = W2.T.dot(dZ2).dot(g1).dot(Z1) #g1为g1'
    dW1 = dZ1.dot(X.T)
    dB1 = dZ1
'''

if __name__ == '__main__':
    main()
    
        