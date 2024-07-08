# # # 生成区间 [0, batch_size] 内的随机整数 
# # import numpy as np
# # #import cupy as cp
# # import scipy as sci
# # import matplotlib.pyplot as plt 
# # from module import Conv2d, Sigmoid, MaxPool2d, AvgPool2d, Linear, ReLU, Tanh, CrossEntropyLoss,flatten
# # import struct
# # import os 
# # import glob
# # import tqdm
# # #import cv2 
# # batch_size = 64
# # random_integer = np.random.randint(0, batch_size) 
# # print(random_integer)
# import numpy as np  
  
# # 假设 'out' 是您的二维数组  
# out = np.array([[1, 2, 3],  
#                 [4, 5, 6],  
#                 [7, 8, 9]])  
  
# # 使用 argmax 函数提取每一行最大数的索引  
# max_indices = np.argmax(out, axis=1)  
# print(max_indices)  
# max_indices=[1,1,6,2,7,1]
# batch_lable=[1,1,1,1,1,1]
# num_matches = 0
# for i in range(6):
#     if(max_indices[i]==batch_lable[i]):
#         num_matches = num_matches + 1
# print(num_matches)
#import cupy as cp
import numpy as np
from module import img2col
# x = cp.arange(6).reshape(2, 3).astype('f')
# print(x, x.sum(axis=1))
 
# with cp.cuda.Device(0):
#    x = cp.array([1, 2, 3, 4, 5])
# print(x.device)
# x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
# print(x.shape)
# print(x[1,1,0])
# x=np.swapaxes(x,0,2)
# print(x[0,1,1])

# 生成1到32的数组
# array0 = np.arange(1, 17).reshape((16,1))
# array1 = np.arange(17, 33).reshape((16,1))
# array = np.concatenate((array0, array1), axis=1)
# # 将一维数组重塑为16行2列的二维数组
# array_2d = array.reshape((16, 2))
# print(array_2d)
# x = array_2d.reshape(4,4,2).transpose(2,0,1)
# print(x)
# [[ 1 17]
#  [ 2 18]
#  [ 3 19]
#  [ 4 20]
#  [ 5 21]
#  [ 6 22]
#  [ 7 23]
#  [ 8 24]
#  [ 9 25]
#  [10 26]
#  [11 27]
#  [12 28]
#  [13 29]
#  [14 30]
#  [15 31]
#  [16 32]]
# [[[ 1  2  3  4]
#   [ 5  6  7  8]
#   [ 9 10 11 12]
#   [13 14 15 16]]

#  [[17 18 19 20]
#   [21 22 23 24]
#   [25 26 27 28]
#   [29 30 31 32]]]
# y = array_2d.reshape(2,4,4)
# print(y)
# x = np.ones(18).reshape((3,3,1,2))
# print(x)
#-----------------验证前向传播---------------
x = np.arange(1, 49).reshape((3,4,4))
print(x)
y = img2col(x,3,1)
print(y)
# k = np.arange(1, 82).reshape((3,3,3,3)) #out in ksize ksize
# print(k)
# kernel_0 = k.reshape(3,-1)
# print(kernel_0)
# kernel = kernel_0.T
# print(kernel)
#---------------验证反向传播--------------
# m = np.arange(8).reshape((2,2,2))
# print(m)
# m1 = np.rot90(m,1,(1,2))
# print(m1)
# m2 = np.rot90(m,1,(0,1))
# print(m2)
# m = np.arange(8).reshape(-1)
# n = np.tile(m,6)
# print(n)
m = np.arange(8).reshape((2,2,2))
print(m.ndim)
print(list(range(10//3+1)))
print("验证集 : epoch= %d时: 损失loss = %.4f, 正确率 = %.4f%%" % (1,2,3))