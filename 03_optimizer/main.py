import urllib.request
import time   
import numpy as np
import pickle
from collections import OrderedDict
import os
from optimizer import *

#---------------------------------------------------------------------------------
#functions.py
#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""   
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
"""    
def cross_entropy_error(y, t):
    delta = 1e-8
    return -np.sum( t * np.log(y + delta))
  
#소프트맥스 함수
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
"""
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
"""    
#미분
def numerical_gradient(f, x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
    return grad

  
#---------------------------------------------------------------------------------
#backpropagation.py
class Relu:
    def __init_(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
      
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
        
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out
        
    def backward(self, dout):     
        dx = np.dot(dout, self.W.T)

        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
       
        
#---------------------------------------------------------------------------------
#LayerNet.py
class TwoLayerNet:
  	#초기 파라미터값 설정
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):     
        self.params = {}
        if 1:
            #가중치 초기값 생성시 작동할부분
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) 
            self.params['b1'] = np.zeros(hidden_size)                                     
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
            self.params['b2'] = np.zeros(output_size)

            #가중치 값 저장하기
            f = open("/projects/mnist_cnn/03_optimizer/weight.pkl", 'wb')
            pickle.dump(self.params, f)
            f.close()
           
  
        #가중치 값 불러오기기
        f = open("/projects/mnist_cnn/03_optimizer/weight.pkl", 'rb')
        self.params = pickle.load(f)
        f.close()

        #계층생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):       #(100,784)                                           
        for layer in self.layers.values():
            x = layer.forward(x)
        return x                #(100,10)
    
    def loss(self,x,t):         #(100,784)(100,10)
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
        
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
       
    def gradient(self,x,t):     #(100,784)(100,10)
        self.loss(x,t)          #순전파
        dout = 1                #역전파
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
                
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw #(784,50)
        grads['b1'] = self.layers['Affine1'].db #(50,)
        grads['W2'] = self.layers['Affine2'].dw #(50,10)
        grads['b2'] = self.layers['Affine2'].db #(10,)     
        return grads
      
      
#---------------------------------------------------------------------------------
#main.py
#초기 mnist.pyURL의 파일다운

mnist_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/dataset/mnist.py"
a=urllib.request.urlopen(mnist_url)
os.chdir('/projects/mnist_cnn/03_optimizer')
k=open("mnist.py","wb")
k.write(a.read())
k.close()
from mnist import load_mnist

#시간 확인 
start_time = time.time() 
tmp_time = start_time

#1.이미지 파일 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False,  one_hot_label=True) #(60000,784)(60000,10)(10000,784)(10000,10)

#3.하이퍼 파라미터 설정
iters_num =  2                   #반복횟수
train_size = x_train.shape[0]   #훈련데이터의 양 60000
batch_size = 1000                 #미니배치 크기 100
learning_rate = 0.001             #학습률

#매개변수 갱신 기법 설정
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

#가중치 초기값 설정
weigh_init_types = {}

#다양한 기법을 담을 딕셔너리(경과기록)
networks = {}  
train_loss_list = {}  
train_acc_list = {}
test_acc_list = {}

#2.클래스 불러오기
for key in optimizers.keys():
    networks[key] = TwoLayerNet(input_size=784, hidden_size=200, output_size=10)   #클래스객체생성
    train_loss_list[key] = []  #딕셔너리안 리스트 생성성
    train_acc_list[key] = []
    test_acc_list[key] = []


#4.학습 시작
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #(100,)
    x_batch = x_train[batch_mask]   #(10000,784)
    t_batch = t_train[batch_mask]   #(10000,10)
    
    #기울기 계산
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        params = networks[key].params
        optimizers[key].update(params, grads)
        
        """
        #매개변수 갱신
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate * grad_backprop[key]
            #print(grad_backprop[key]) #Eureka!
        """
        #학습 경과 기록
        loss = networks[key].loss(x_batch, t_batch)
        train_acc = networks[key].accuracy(x_train, t_train)
        test_acc = networks[key].accuracy(x_test, t_test)
    
        train_loss_list[key].append(round(loss,2)) #리스트에 요소 추가
        train_acc_list[key].append(round(train_acc,2))
        test_acc_list[key].append(round(test_acc,2))
        
 
        train_loss_list[key] = list(map(int, train_loss_list[key])) #리스트내용을 int로 변환     
        
        train_loss_list[key][i] = round((train_loss_list[key][i] / batch_size),3)
        train_acc_list[key][i] = round(train_acc_list[key][i] ,3)
        test_acc_list[key][i] = round(test_acc_list[key][i] ,3)
        """
        print(str(i+1) + " loss      : " + str(round((train_loss_list[key][i] / batch_size),4)))
        print(str(i+1) + " train acc : " + str(round(train_acc,4)))
        print(str(i+1) + " test acc  : " + str(round(test_acc,4)))
        
        #1회 학습에 걸린시간
        mid_time = time.time()
        one = mid_time - tmp_time
        print(str(i+1) + " time      : " + str(round(one,2)) + "초")
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        tmp_time = mid_time    
        """    
        
for key in optimizers.keys():
    print(key)
    print(" loss      : " + str(train_loss_list[key]))
    print(" train acc : " + str(train_acc_list[key]))
    print(" train acc : " + str(test_acc_list[key]))
    
    #가중치 값 저장하기
    f = open("/projects/mnist_cnn/03_optimizer/weight_" + key + ".pkl", 'wb')
    pickle.dump(networks[key].params, f)
    f.close()

#5.총 걸린시간    
end_time = time.time()
print("Running time : " + str(int(end_time-start_time)) + "초")