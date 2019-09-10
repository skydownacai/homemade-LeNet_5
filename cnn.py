from minst_read import *
import matplotlib.pyplot as pyplot
import numpy as np
train_x = load_train_images() / 255
train_y = load_train_labels()
test_x  = load_test_images() / 255
test_y  = load_test_labels()
def relu(x):
    return np.max(0, x)
x = np.random.rand(3,2,2)
w,h,d = x.shape
class LeNet_5:
    '''structure:
         image  -> (28*28*1)
         conv(filter 3x3x1,stride 1,total 16) -> (26 * 26 * 16)
         max pool -> (13 * 13 * 16)
         conv(filter 4x4x16,stride 1,total 32) -> (10 * 10 * 32)
         max pool -> (5 * 5 * 32)

         conv(filter 3x3x32,stride 1,total 10)-> ( 3  * 3  * 10)
         average pool -> (1 * 1 * 10)
         softmax -> (1 * 1 *10) -> crossentrophy : Loss_function
         predict  = argmax (output,axis = '2')
       optimizer:
         mini-batch SGD
       cost-function:
         l2
    '''
    def __init__(self,x):
        self.filters = [
            [np.random.randn(3, 3, x.shape[2])/100  for k in range(16)],
            [np.random.randn(4, 4, 16)/100 for k in range(32)],
            [np.random.randn(3, 3, 32)/100 for k in range(10)]
        ]
        self.max_pool_size = [2,2]
        self.max_pool_strides = [1,1]
        self.filter_nums = [16,32,10]
        self.strides = [1,1,1]
        self.maxpool_index = [0,0]
        self.bias = [np.random.randn(num,1) for num in self.filter_nums]
        self.learning_rate = 0.01
        self.convs = [0,0,0]
        self.relus = [0,0,0]
        self.learning_rate = 0.001
        self.pools = [0,0,0]
        self.history = {'accu':[],'loss':[]} #用于记录训练信息
        self.y_p =np.zeros((10,1))
        self.W1 = 0
        self.W2 = 0
    def relu(self,x):
        after_relu = np.zeros(x.shape)
        w, h, d = x.shape
        for i in range(w):
            for j in range(h):
                for c in range(d):
                    after_relu[i][j][c] = max(0,x[i][j][c])
        return after_relu
    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x))
    def cross_entropy(self,x,y):
        '''x是softmax之后的值,y是真实的值'''
        y_true = np.zeros((10,1))
        y_true[int(y)] = 1
        return -1 * (np.log(x).T @ y_true + np.log(1-x).T @ (1 - y_true))
    def max_pool(self,x,layer,F = 2,s = 2):
        '''spatial extent:2x2(as the paramter F stride:2(as the parameter s) '''
        w, h, d = x.shape
        after_max_pool = np.zeros((int(1 + (w - F)/s),int(1 + (h - F)/s),d),dtype=np.float64)
        self.maxpool_index[layer] = np.zeros(after_max_pool.shape)
        for c in range(d):
            for i in range(after_max_pool.shape[0]):
                for j in range(after_max_pool.shape[1]):
                    pool_spatial = x[i * s : i * s + F,j * s : j * s + F,c]
                    after_max_pool[i][j][c] = np.max(pool_spatial)
                    arg_max = np.argmax(pool_spatial)
                    self.maxpool_index[layer][i][j][c] = arg_max
        return after_max_pool
    def average_pool(self,x,F,s):
        w, h, d = x.shape
        after_max_pool = np.zeros((int(1 + (w - F)/s),int(1 + (h - F)/s),d),dtype=np.float64)
        for c in range(d):
            for i in range(after_max_pool.shape[0]):
                for j in range(after_max_pool.shape[1]):
                    pool_spatial = x[i * s : i * s + F,j * s : j * s + F,c]
                    after_max_pool[i][j][c] = np.average(pool_spatial)
        return after_max_pool
    def conv(self,x,filters,bias,s = 1,pad = 0):
        ''' 对图像进行长度为pad的zero填充
           传入一个滤波器组,然后进行步长为s的卷积
           一组滤波器的spatial extent是同样的
        '''
        F = filters[0].shape[0]
        #spatial extent of filter
        after_conv = np.zeros(
            (int(1 + (x.shape[0] - F) / s),
             int(1 + (x.shape[1] - F) / s),
             len(filters))
        )
        for c in range(len(filters)):
            filter = filters[c]
            for i in range(after_conv.shape[0]):
                for j in range(after_conv.shape[1]):
                    filter_space = x[i * s : i * s + F ,j * s : j * s + F,:]
                    inner_dot = filter_space * filter
                    product = np.sum(inner_dot)
                    after_conv[i][j][c] = product + bias[c][0]
        return after_conv
    def visual(self,x,title):
        x = x[:,:,0].reshape(x.shape[0],x.shape[0])
        pyplot.imshow(x)
        plt.title(title)
        plt.show()
    def fit(self,X,Y,batch):
        '''mini-batch SGD'''
        for iteration in range(1000):
            mini_batch = np.random.randint(0,len(X),batch)
            for index in mini_batch:
                x = X[index].reshape(28,28,1)
                y = Y[index]
                self.forward(x)
                print('yp输出:',self.y_p[:,0])
                # 前向传播
                dbias, dfilters = self.gradient_of_network(x, y)
                # 后向传播计算梯度
                for i in range(3):
                    for j in range(len(dfilters[i])):
                        self.filters[i][j] -= self.learning_rate * dfilters[i][j]/batch
                    self.bias[i] -= self.learning_rate * dbias[i]/batch
                y_true = np.zeros((10, 1))
                y_true[int(y)] = 1
                #print(y_true)
            loss,accu = self.metrics(X,Y)
            print('iteration:',iteration,' loss:',loss,' accu',accu)
    def metrics(self,X,Y):
        '''计算所有样本的平均loss,与精度'''
        loss = []
        accu = []
        for i in range(len(X)):
            x = X[i].reshape(28,28,1)
            y = Y[i]
            self.forward(x)
            loss.append(self.cross_entropy(self.y_p,y))
            accu.append(np.argmax(self.y_p) == y)
        return np.average(loss),np.average(accu)
    def gradient_of_network(self,x,y):
        '这里每一层都是关于输出的导数，而不是关于输入的'
        y_true = np.zeros((10,1))
        y_true[int(y)] = 1
        dpools2 = (np.diag(self.y_p[:,0]) - self.y_p @ self.y_p.T) @ (-1*y_true/self.y_p + (1 - y_true)/(1 - self.y_p))
        #print('dL/dv  :',((np.diag(self.y_p[:,0]) - self.y_p @ self.y_p.T) @ (-1*y_true/self.y_p + (1 - y_true)/(1 - self.y_p)))[:,0])

        drelus2 = np.zeros(self.relus[2].shape)
        for i in range(self.relus[2].shape[2]):
            drelus2[:,:,i] = dpools2[i][0]/9

        dconv2  = np.zeros(self.convs[2].shape)
        w, h, d = dconv2.shape
        for i in range(w):
            for j in range(h):
                for c in range(d):
                    dconv2[i][j][c] = drelus2[i][j][c]*(self.convs[2][i][j][c] > 0)

        #print('dconvs2\n',dconv2[:,:,0])
        dbias2 = np.sum(np.sum(dconv2,axis=0),axis=0).reshape(self.bias[2].shape)

        # 计算bias的导数
        dfilters2 = [np.zeros(self.filters[2][0].shape) for i in range(self.filter_nums[2])]
        dpools1 = np.zeros(self.pools[1].shape)
        for c in range(self.filter_nums[2]):
            f_w, f_h, f_d = dfilters2[c].shape
            f_stride = self.strides[2]
            for i in range(w):
                for j in range(h):
                    dfilters2[c] += dconv2[i][j][c] * self.pools[1][i*f_stride : i*f_stride + f_w ,j*f_stride:j * f_stride + f_h,:]
                    dpools1[i*f_stride : i*f_stride + f_w ,j*f_stride:j * f_stride + f_h,:] += self.filters[2][c] * dconv2[i][j][c]
        #计算滤波器的导数 与 pools的导数
        drelus1 = np.zeros(self.relus[1].shape)

        for c in range(self.relus[1].shape[2]):
            for i in range(dpools1.shape[0]):
                for j in range(dpools1.shape[1]):
                    F =  self.max_pool_size[1]
                    s =  self.max_pool_strides[1]
                    argmax = self.maxpool_index[1][i][j][c]
                    drelus1[i * s + int( argmax / F),j * s + int(argmax % F),c] = dpools1[i][j][c]
        #计算relus1的导数

        dconv1  = np.zeros(self.convs[1].shape)
        w, h, d = dconv1.shape
        for i in range(w):
            for j in range(h):
                for c in range(d):
                    dconv1[i][j][c] = drelus1[i][j][c]*(self.convs[1][i][j][c] > 0)

        dbias1 = np.sum(np.sum(dconv1,axis=0),axis=0).reshape(self.bias[1].shape)
        # 计算bias的导数

        dfilters1 = [np.zeros(self.filters[1][0].shape) for i in range(self.filter_nums[1])]
        dpools0 = np.zeros(self.pools[0].shape)
        for c in range(self.filter_nums[1]):
            f_w, f_h, f_d = dfilters1[c].shape
            f_stride = self.strides[1]
            for i in range(w):
                for j in range(h):
                    dfilters1[c] += dconv1[i][j][c] * self.pools[0][i*f_stride : i*f_stride + f_w ,j*f_stride:j * f_stride + f_h,:]
                    dpools0[i*f_stride : i*f_stride + f_w ,j*f_stride:j * f_stride + f_h,:] += self.filters[1][c] * dconv1[i][j][c]
        #计算滤波器的导数 与 pools的导数

        drelus0 = np.zeros(self.relus[0].shape)
        for c in range(self.relus[0].shape[2]):
            for i in range(dpools1.shape[0]):
                for j in range(dpools1.shape[1]):
                    F =  self.max_pool_size[0]
                    s =  self.max_pool_strides[0]
                    argmax = self.maxpool_index[0][i][j][c]
                    drelus0[i * s + int( argmax / F),j * s + int(argmax % F),c] = dpools0[i][j][c]

        dconv0  = np.zeros(self.convs[0].shape)
        w, h, d = dconv0.shape
        for i in range(w):
            for j in range(h):
                for c in range(d):
                    dconv0[i][j][c] = drelus0[i][j][c]*(self.convs[0][i][j][c] > 0)

        dbias0 = np.sum(np.sum(dconv0,axis=0),axis=0).reshape(self.bias[0].shape)
        # 计算bias的导数

        dfilters0 = [np.zeros(self.filters[0][0].shape) for i in range(self.filter_nums[0])]
        for c in range(self.filter_nums[0]):
            f_w, f_h, f_d = dfilters0[c].shape
            f_stride = self.strides[0]
            for i in range(w):
                for j in range(h):
                    dfilters0[c] += dconv0[i][j][c] * x[i*f_stride : i*f_stride + f_w ,j*f_stride:j * f_stride + f_h,:]

        #计算滤波器的导数 与 pools的导数

        return [dbias0,dbias1,dbias2],[dfilters0,dfilters1,dfilters2]
    def bp(self):
        pass
    def forward(self,x):
        '''输入一个x  计算值输出'''
        self.convs[0] = self.conv(x,self.filters[0],self.bias[0])
        self.relus[0] = self.relu(self.convs[0])
        self.pools[0]  = self.max_pool(self.relus[0],0)

        self.convs[1] = self.conv(self.pools[0],self.filters[1],self.bias[1])
        self.relus[1] = self.relu(self.convs[1])
        self.pools[1]  = self.max_pool(self.relus[1],1)

        self.convs[2] = self.conv(self.pools[1],self.filters[2],self.bias[2])
        self.relus[2] = self.relu(self.convs[2])
        self.pools[2]  = self.average_pool(self.relus[2],3,3)

        self.y_p = self.softmax(self.pools[2].reshape(-1,1))
    def predict(self,x):
        return np.argmax(self.y_p)
pyplot.imshow(train_x[0])
plt.title('original picture')
plt.show()
x = train_x[0].reshape(28,28,1)
y = train_y[0]
a = LeNet_5(x)
a.fit(train_x[:1],train_y[:1],1)