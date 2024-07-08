import numpy as np

class Con2d():
    """
      卷积前向：
      输入:input:[b, cin, h, w]
           weight:[cin, cout, ksize, ksize], stride, padding 
     计算过程：
         1.  将权重拉平成：[cout, cin*ksize*ksize] self.weight 先transpose(1, 0, 2,3) 再reshpe(cout, -1)
         2.  将输入整理成：[b*hout*wout,cin*ksize*ksize]: 
             先根据hin和win 通过pad, ksize和stride计算出hout和wout (h+2*pad-ksize)//stride + 1 (b, cout, hout, wout)
             再根据img展平 整理成自己的 img  (b, hout, wout, cin*kszie*ksize)  -> (b*hout*wout, cin*kszie*ksize)
         3. 两者相乘后 np.dot 再去reshape (cout, b*hout*wout) -> (b, cout, hout*wout)
    """
    """
     卷积反向：
     输入 input:[b, cout, hout, wout] -loss 
     计算过程： 
         1. 将输入换成输出格式： [b, cout, hout, wout] -> [cout, b, hout, wout] ->[cout, b*hout*wout] 
         2. 计算的输入与之前的图相乘： (cout, b*hout*wout) * (b*hout*wout, cin*kszie*ksize) -> (cout, cin*kszie*ksize) 得到更新后的权重
         3. 将更新后的权重与图相乘，
 
    """
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride=1 ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ksize = kernel_size
        self.padding = padding
        self.stride = stride

        self.weights = np.random.standard_normal((out_channel, in_channel, kernel_size, kernel_size))
        self.bias = np.zeros(out_channel)
        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.bias.shape)

    def img2col(self, x, ksize, strid):
        b,c,h,w = x.shape # (5, 3, 34, 34)
        img_col = []
        for n in range(b): # 5
            for i in range(0, h-ksize+1, strid):
                for j in range(0, w-ksize+1, strid):
                        col = x[n,:, i:i+ksize, j:j+ksize].reshape(-1) # (1, 3, 4, 4) # 48
                        img_col.append(col)
        return np.array(img_col) # (5, 3, 31, 31, 48)
 
    def forward(self, x):
        self.x = x #(5, 3, 34,34)
        weights = self.weights.reshape(self.out_channel, -1) # (12, 3*4*4)
        x = np.pad (x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), "constant") # (5, 3, 34, 34)
        b, c, h, w = x.shape
        self.out = np.zeros((b, self.out_channel, (h-self.ksize)//self.stride+1, (w-self.ksize)//self.stride+1))# (5, 12, 31, 31)
        self.img_col = self.img2col(x, self.ksize, self.stride) #  (5, 31, 31, 48) #(4805, 48)
        out = np.dot(weights, self.img_col.T).reshape(self.out_channel, b, -1).transpose(1, 0,2) # (12 ,48) *(48, 4805) = (12, 4805) =(12, 5, 961) =(5, 12, 961)
        self.out = np.reshape(out, self.out.shape) 
        return self.out
 
    def backward(self, grad_out):
        b, c, h, w = self.out.shape
        grad_out_ = grad_out.transpose(1, 0, 2, 3 )
        grad_out_flag = np.reshape(grad_out_,[self.out_channel, -1]) # [cout, b*h*w]
        self.grad_w = np.dot(grad_out_flag, self.img_col).reshape(c, self.in_channel, self.ksize, self.ksize) #  (cout, cin*kszie*ksize)  -权重值
        self.grad_b = np.sum(grad_out_flag, axis=1) # [cout] -偏置值
        tmp = self.ksize -self.padding -1
        grad_out_pad = np.pad(grad_out, ((0,0),(0,0),(tmp, tmp),(tmp,tmp)),'constant')
        weights = self.weights.transpose(1, 0, 2, 3).reshape([self.in_channel, -1]) # [cin. cout*ksize*ksize]
        col_grad = self.img2col(grad_out_pad, self.ksize, 1) # 
        next_eta = np.dot(weights, col_grad.T).reshape(self.in_channel, b, -1).transpose(1, 0, 2)
        next_eta = np.reshape(next_eta, self.x.shape)
        return next_eta

    def zero_grad(self):
        self.grad_w = np.zeros_like(self.grad_w)  
        self.grad_b = np.zeros_like(self.grad_b)

    def update(self, lr=1e-3):
        self.weights -= lr*self.grad_w
        self.bias -= lr*self.grad_b 

if __name__ == '__main__':
    x = np.ones([5,3,34,34])
    conv = Con2d(3,12,3,1,1)
    print(conv.img2col(x,3,1).shape)
    # for i in range(100):
    #     y = conv.forward(x)
    #     loss =abs( y - 1)
    #     x = conv.backward(loss)
    #     lr = 1e-4 
    #     conv.update(lr)
    #     print(np.sum(loss))
