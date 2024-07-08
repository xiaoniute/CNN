import numpy as np
#import cupy as cp
import scipy as sci
import matplotlib.pyplot as plt 
from module import Conv2d, Sigmoid, MaxPool2d, AvgPool2d, Linear, ReLU, CrossEntropyLoss,flatten
import struct
import os 
import glob
from tqdm import tqdm
import time 
#import cv2

def load_mnist(path, kind='train'):

    image_path = glob.glob('./**/%s*3-ubyte' % (kind),recursive=True)[0]
    label_path = glob.glob('./**/%s*1-ubyte' % (kind),recursive=True)[0]

    with open(label_path, "rb") as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, "rb") as impath:
        magic, num, rows, cols = struct.unpack('>IIII', impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 28*28)

    return images, labels

class LeNet5:
    def __init__(self):
        #输入(N,1,28,28)
        self.conv1 = Conv2d(1, 6, 5, 1, 2) #输出(N,6,28,28)
        self.relu1 = Sigmoid()
        self.pool1 = AvgPool2d(2)         #输出(N,6,14,14)
        self.conv2 = Conv2d(6, 16, 5)     #输出(N,16,10,10)
        self.relu2 = Sigmoid()
        self.pool2 = AvgPool2d(2)        #输出(N,16,5,5)
        self.flat = flatten()
        self.fc1 = Linear(16*5*5, 120) #输出(N,120)
        self.relu3 = Sigmoid()
        self.fc2 = Linear(120, 84) #输出(N,84)
        self.relu4 = Sigmoid()
        self.fc3 = Linear(84, 10) #输出(N,10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flat.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.relu4.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, dy, lr):
        dy = self.fc3.backward(dy, lr)
        dy = self.relu4.backward(dy)
        dy = self.fc2.backward(dy, lr)
        dy = self.relu3.backward(dy)
        dy = self.fc1.backward(dy, lr)
        dy = self.flat.backward(dy)
        dy = self.pool2.backward(dy)
        dy = self.relu2.backward(dy)
        dy = self.conv2.backward(dy, lr)
        dy = self.pool1.backward(dy)
        dy = self.relu1.backward(dy)
        dy = self.conv1.backward(dy, lr)

if __name__ == '__main__':

    #define hyperparameters
    lr = 0.002
    epoch = 50
    batch_size = 64
    #-------------------------数据处理-----------------------------------
    train_images, train_labels = load_mnist("mnist_dataset", kind="train")
    test_images, test_labels = load_mnist("mnist_dataset", kind="t10k")

    train_images = train_images.astype(np.float16) / 256
    test_images = test_images.astype(np.float16) / 256

    train_images = train_images.reshape(-1,1,28,28)
    test_images  = test_images.reshape(-1,1,28,28)
    # ----------------------------定义网络----------------------------
    net = LeNet5()
    lossfunction = CrossEntropyLoss() 
    # ----------------------------绘图----------------------------
    # plt.imshow(train_images[0][0], cmap='gray')  
    # plt.axis('off')  # 关闭坐标轴  
    # plt.show()  
    # print(train_images[0][0].shape) #输出[28,28]
    # print(train_images[0][0])
    # 注释掉归一化后验证成功， 不注释也可以
    # ------------------------训练流程-------------------------
    loss_col = []
    accuracy_col = []
    start =  time.perf_counter()

    for _ in range(epoch):
        # #--------------训练-----------
        N = train_images.shape[0] #训练样本总数 
        tq = tqdm(list(range(N//batch_size+1)))
        loss_sum = 0
        for i in tq:
            tq.set_description("Epoch %d" % _)
            if (i+1)*batch_size<N:
                # 生成区间 [i*batch_size, (i+1)*batch_size-1] 内的随机整数  
                random_integer = np.random.randint(i*batch_size, (i+1)*batch_size,1)  
            else:
                random_integer = np.random.randint(i*batch_size, N,1)

                # batch_images = train_images[i*batch_size : (i+1)*batch_size]
                # batch_lable = train_labels[i*batch_size : (i+1)*batch_size]
                
            batch_images = train_images[random_integer]
            batch_lable = train_labels[random_integer]
             
            out = net.forward(batch_images)
            loss, dx = lossfunction(out , batch_lable)

            loss_sum +=loss

            dx = net.backward(dx , lr)
        NUM = N//batch_size+1 
        loss = loss_sum/NUM
        print("训练集 : epoch=%d时: 损失loss = %.4f" % (_,loss))
        #--------------------验证---------------------
        random = np.random.randint(0, test_images.shape[0], batch_size) 
        batch_images = test_images[random]
        batch_lable = test_labels[random]
        out = net.forward(batch_images)
        loss,dx = lossfunction(out , batch_lable)
        max_indices = np.argmax(out, axis=1)
        num_matches = 0
        for i in range(batch_size):
            if(max_indices[i]==batch_lable[i]):
                num_matches = num_matches + 1
        accuracy = num_matches/(batch_size)

        loss_col.append(loss)
        accuracy_col.append(accuracy) 

        print("验证集 : epoch=%d时: 损失loss = %.4f, 正确率 = %.4f%%" % (_,loss,accuracy*100))

        # if loss<0.5:
        #     lr = 0.001
    end = time.perf_counter()
    print("运行时间：%ds"% (end-start) )
    #------------------------测试---------------------------------
    # print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
    # print(train_images.shape)
    # # print(train_images[0])
    # arr = np.random.rand(10,1,28,28)
    # out = net.forward(arr)
    # #loss = CrossEntropyLoss(out , )
    # #print(out.shape) 
    # x = cp.arange(6).reshape(2, 3).astype('f')
    # print(x, x.sum(axis=1))
    # ---------------------------------------------------------------------------
    #--------------------绘图----------------------------
    # 创建一个figure和一个axes
    fig, ax1 = plt.subplots()
    x = list(range(epoch))

    # 绘制第一个y轴的数据
    ax1.plot(x, loss_col, 'g-', label='loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='g')

    # 绘制第二个y轴的数据
    ax2 = ax1.twinx()
    ax2.plot(x, accuracy_col, 'b-', label='accuracy')
    ax2.set_ylabel('accuracy', color='b')

    # 设置图表标题
    plt.title('test results')

    # 显示图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 显示图表
    plt.show()
    

    
            
            

