import cupy as cp

def img2col(x,ksize,stride):
    if x.ndim != 3:
        x = x[cp.newaxis,:,:]
    C,H,W = x.shape
    # 计算输出的大小  
    H_out = (H - ksize) // stride + 1  
    W_out = (W - ksize) // stride + 1
    image_col = cp.zeros((H_out*W_out,ksize*ksize*C))
    num = 0
    for i in range(H_out):
        for j in range(W_out): 
            image_col[num] =  x[:,i*stride:i*stride+ksize, j*stride:j*stride+ksize].reshape(-1)
            num += 1

    return image_col

class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dtype = None):
        
        self.x = None
        # 初始化权重和偏置  
        self.weight = cp.random.normal(loc=0.0, scale=1.0, size=(out_channels, in_channels, kernel_size, kernel_size))  
        self.bias = cp.zeros((out_channels, 1))  
            
        # 设置其他超参数  
        self.stride = stride  
        self.padding = padding  
        self.dtype = dtype

        #初始化梯度
        self.w_grad = cp.zeros(shape=(out_channels, in_channels, kernel_size, kernel_size))
        self.b_grad = cp.zeros((out_channels, 1))
    
    def forward(self, x):
        """
        x - shape (N, C, H, W)
        weight - shape (O, C , WH, WW)
        return the result of Conv2d with shape (N, O, H', W')
        """
        self.x = x
        padding = self.padding
        # padding的数量  
        if self.padding != 0:
            pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))  # 在最后两个维度的四周各添加一行  
            # 常数填充  
            self.x = cp.pad(x, pad_width, mode='constant', constant_values=0)

        N, C, H, W = x.shape  
        O, C, WH, WW = self.weight.shape  
        stride = self.stride  
        padding = self.padding  
    
        # 计算输出的大小  
        H_out = (H + 2 * padding - WH) // stride + 1  
        W_out = (W + 2 * padding - WW) // stride + 1  
    
        # 初始化输出  
        out = cp.zeros((N, O, H_out, W_out))  

        # 执行卷积操作  
        self.image_col = []
        kernel_0 = self.weight.reshape(O,-1)
        kernel = kernel_0.T

        for i in range(N):
            image_col = img2col(self.x[i],WH,self.stride)
            out[i] = (cp.dot(image_col,kernel)+self.bias.T).reshape(H_out,W_out,O).transpose(2,0,1) #这里利用1-32的，16*2的数组进行了验证
            self.image_col.append(image_col)        

        # for i in range(N):  # 对每个样本  
        #     for f in range(F):  # 对每个输出通道  
        #         for j in range(H_out):  # 对输出的高  
        #             for k in range(W_out):  # 对输出的宽  
        #                 # 计算当前窗口下的卷积
        #                 # window 各个维度解释：  
        #                 # 1  i - 输入的样本序号
        #                 # 2  样本的每个输入通道都要选取
        #                 # 3  第三个维度 高度区域
        #                 # 4  第四个维度 宽度区域
        #                 window = x[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW]  
        #                 # 选取卷积核计算解释：
        #                 # 每个输出通道对应一个卷积核 
        #                 out[i, f, j, k] = cp.sum(window * self.weight[f, :, :, :]) + self.bias[f]  

        return out

    def backward(self, dy, lr):
        """
        dy - the gradient of last layer with shape (N, O, H_OUT, W_OUT)
        lr - learning rate
        calculate self.w_grad to update self.weight,
        calculate self.b_grad to update self.bias,
        return the result of gradient dx with shape (N, C, H, W)
        """
        N, O, H_out, W_out = dy.shape  
        N, C, H, W = self.x.shape  
        O, C, WH, WW = self.weight.shape  
        stride = self.stride  

        # 计算self.w_grad(F,C,WH,WW),self.b_grad
        #初始化梯度
        self.w_grad = cp.zeros(shape=(O, C, WH, WW))
        self.b_grad = cp.zeros((O, 1))

        for j in range(O):

            for i in range(N):

                delta_kernel_0 = dy[i][j].reshape(-1)
                # delta_kernel_1 = cp.tile(delta_kernel_0,C)
                # delta_kernel = delta_kernel_1[cp.newaxis,:].T
                delta_kernel = delta_kernel_0.T

                for k in range(C):
                    image_col = img2col(self.x[i][k],H_out,stride)
                    self.w_grad[j][k] += cp.dot(image_col, delta_kernel).reshape(WH,WW)

                self.b_grad[j] += cp.sum(delta_kernel)

            self.w_grad[j] /= N
            self.b_grad[j] /= N

        #计算dx
        dx = cp.zeros(self.x.shape)

        k_180 = cp.rot90(self.weight, 2, (2,3))      # numpy矩阵旋转180度
        #填充dy
        pad = WH-1
        pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))  # 在最后两个维度的四周各添加一行  
        pad_dy = cp.pad(dy, pad_width, mode='constant', constant_values=0)

        for i in range(N):
            delta_col = img2col(pad_dy[i],WH,self.stride)
            kernel = k_180.transpose(1,0,2,3).reshape(C,-1).T
            dx[i] += cp.dot(delta_col, kernel).reshape(H,W,C).transpose(2,0,1)
        dx /= N


        self.weight -=  self.w_grad * lr
        self.bias -=  self.b_grad * lr

        # for i in range(N):  # 对每个样本  
        #     for f in range(F):  # 对每个输出通道  
        #         for j in range(H_out):  # 对输出的高  
        #             for k in range(W_out):  # 对输出的宽  
        #                 # 计算当前窗口下的卷积  
        #                 window = x[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW]  
        #                 dw[f, :, :, :] += dy[i, f, j, k] * window  
        #                 db[f] += dy[i, f, j, k]  
        #                 dx_padding[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW] += dy[i, f, j, k] * self.weight[f, :, :, :]  

        # # 更新权重和偏置  
        # self.weight -= lr * dw  
        # self.bias -= lr * db  
        # if padding!= 0:
        #     dx = dx_padding[:,:,padding:-padding,padding:-padding] 
        # else:
        #     dx = dx_padding

        return dx
    
     
class ReLU:
    def forward(self, x):
        self.x = x
        return cp.maximum(0, x)
    def backward(self, dy):
        if self.x<0:
            return 0
        else:
            return dy

# class Tanh:
#     def forward(self, x):
#         return cp.tanh(x)
#     def backward(self, dy):
#         return dy * (1 - cp.tanh(self.forward(dy)) ** 2)  

class Sigmoid:
    def forward(self, x):
       self.y = 1 / (1 + cp.exp(-x)) 
       return self.y 
    def backward(self, dy):
       return dy * self.y * (1 - self.y)
    

       
class MaxPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):
        self.kernel_size = kernel_size  
        self.stride = stride if stride is not None else kernel_size  
        self.padding = padding  
        self.x = None  # used to store input for backward pass

    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of MaxPool2d with shape (N, C, H', W')
        """
        self.x = x  # save input for use in backward pass  
        N, C, H, W = x.shape  
        HH, WW = self.kernel_size, self.kernel_size  
        stride = self.stride  
        padding = self.padding  
  
        H_out = 1 + (H + 2 * padding - HH) // stride  
        W_out = 1 + (W + 2 * padding - WW) // stride  
  
        out = cp.zeros((N, C, H_out, W_out))  

        # padding的数量  
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))  # 在最后两个维度的四周各添加一行  
        
        # 常数填充  
        x = cp.pad(x, pad_width, mode='constant', constant_values=0)

        for i in range(N):  
            for j in range(C):  
                for k in range(H_out):  
                    for l in range(W_out):  
                        window = x[i, j, k * stride:k * stride + HH, l * stride:l * stride + WW]  
                        out[i, j, k, l] = cp.max(window)  
  
        return out

    def backward(self, dy):
        """
        dy - shape (N, C, H', W')
        return the result of gradient dx with shape (N, C, H, W)
        """
        N, C, H, W = self.x.shape  
        HH, WW = self.kernel_size, self.kernel_size  
        stride = self.stride  
        padding = self.padding  
        dx = cp.zeros_like(self.x)  

        # padding的数量  
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))  # 在最后两个维度的四周各添加一行  
        
        # 常数填充  
        x = cp.pad(x, pad_width, mode='constant', constant_values=0)

        for i in range(N):  
            for j in range(C):  
                for k in range(dy.shape[2]):  
                    for l in range(dy.shape[3]):  
                        window = self.x[i, j, k * stride:k * stride + HH, l * stride:l * stride + WW]  
                        mask = (window == cp.max(window))  
                        dx[i, j, k * stride:k * stride + HH, l * stride:l * stride + WW] += mask * dy[i, j, k, l]  
  
        return dx
    
class AvgPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):

        self.kernel_size = kernel_size  
        self.stride = stride if stride is not None else kernel_size  
        self.padding = padding  
        self.x = None  # used to store input for backward pass

    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of AvgPool2d with shape (N, C, H', W')
        """
        self.x = x  # save input for use in backward pass  
        N, C, H, W = x.shape  
        WH, WW = self.kernel_size, self.kernel_size  
        stride = self.stride  
        padding = self.padding  
  
        H_out = 1 + (H + 2 * padding - WH) // stride  
        W_out = 1 + (W + 2 * padding - WW) // stride  
  
        out = cp.zeros((N, C, H_out, W_out))  

        # padding的数量  
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))  # 在最后两个维度的四周添加 
        
        # 常数填充  
        self.x = cp.pad(x, pad_width, mode='constant', constant_values=0)
    
        for i in range(N):  
            for j in range(C):  
                for k in range(H_out):  
                    for l in range(W_out):  
                        window = x[i, j, k * stride:k * stride + WH, l * stride:l * stride + WW]  
                        out[i, j, k, l] = cp.mean(window)  
  
        return out

    def backward(self, dy):
        """
        dy - shape (N, C, H', W')
        return the result of gradient dx with shape (N, C, H, W)
        """
        N, C, H, W = self.x.shape  
        N, C, H_OUT, W_OUT = dy.shape
        WH, WW = self.kernel_size, self.kernel_size  
        stride = self.stride  
        padding = self.padding  

        dx_padding = cp.zeros_like(self.x)  

        for i in range(N):  
            for j in range(C):  
                for k in range(H_OUT):  
                    for l in range(W_OUT):  
                        dy_val = dy[i, j, k, l]  / (WH * WW)
                        dx_padding[i, j, k * stride:k * stride + WH, l * stride:l * stride + WW] += dy_val  

        if padding!= 0:
            dx = dx_padding[:,:,padding:-padding,padding:-padding] 
        else:
            dx = dx_padding
        return dx
        
class flatten:  
    def forward(self, x):  
        self.shape = x.shape  
        #展平除了第一维的其他三维
        flattened_length = self.shape[1] * self.shape[2] * self.shape[3]  
        flattened_array = x.reshape(self.shape[0], flattened_length)  
        return flattened_array  
      
    def backward(self, dy):  
        dx = dy.reshape(self.shape)  # 重新排列梯度的形状  
        return dx  

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):

        self.in_features = in_features  
        self.out_features = out_features  
        self.weight = cp.random.randn(in_features, out_features)  
        self.bias = cp.zeros((1, out_features)) if bias else None  
        self.x = None  # used to store input for backward pass 
        
    def forward(self, x):
        """
        x - shape (N, C)
        return the result of Linear layer with shape (N, O)
        """
        self.x = x  # save input for use in backward pass  
        output = cp.dot(x, self.weight)  
        if self.bias is not None:  
            output += self.bias  
        return output


    def backward(self, dy, lr):
        """
        dy - shape (N, O)
        return the result of gradient dx with shape (N, C)
        """
        N , O = dy.shape
        dx = cp.dot(dy, self.weight.T)
        dw = cp.dot(self.x.T, dy)/N
        self.weight -= lr * dw  
        if self.bias is not None:  
            db = cp.sum(dy, axis=0, keepdims=True)/N  
            self.bias -= lr * db  
        return dx

class CrossEntropyLoss:
    def __call__(self, x, label):
        N = x.shape[0]  
        #减去每行的最大值，避免指数运算时出现数值溢出
        exp_scores = cp.exp(x - cp.max(x, axis=1, keepdims=True))  
        probs = exp_scores / cp.sum(exp_scores, axis=1, keepdims=True)  
        correct_logprobs = -cp.log(probs[range(N), label])  
        data_loss = cp.sum(correct_logprobs) / N  
        # 反向传播  
        dx = probs.copy()  
        dx[range(N), label] -= 1  
          
        return data_loss, dx
        