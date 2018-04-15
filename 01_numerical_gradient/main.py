import urllib.request
import time   
import numpy as np
import pickle
import os

#---------------------------------------------------------------------------------
#functions.py
#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
#소프트맥스 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    
#교차 엔트로피 에러
def cross_entropy_error(y,t):
    delta = 1e-7 #0.0000001
    return -np.sum(t * np.log(y + delta))

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
#LayerNet.py
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):     #초기 파라미터값 설정
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  #(784,50)
        self.params['b1'] = np.zeros(hidden_size)                                       #(50,)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)  #(50,10)
        self.params['b2'] = np.zeros(output_size)                                       #(10,)
    
    def predict(self, x):       #(100,784)                                              #이미지 예측 
        W1,W2 = self.params['W1'], self.params['W2']                 
        b1,b2 = self.params['b1'], self.params['b2']                                    #활성화 함수
        a1 = np.dot(x, W1) + b1                                                         #A = XW + B    
        z1 = sigmoid(a1)                                                                #sigmoid()
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y                #(100,10)
    
    def loss(self,x,t):         #(100,784)(100,10)
        y = self.predict(x)
        return cross_entropy_error(y,t)
        
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self,x,t):     #(100,784)(100,10)
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #(784,50)
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) #(50,)
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) #(50,10)
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) #(10,)     
        return grads
#---------------------------------------------------------------------------------
#main.py
#URL의 파일다운
mnist_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/dataset/mnist.py"
a=urllib.request.urlopen(mnist_url)
os.chdir('mnist_c/mnist_cnn/01_numerical_gradient')
k=open("mnist.py","wb")
k.write(a.read())
k.close()

#시간 확인 
start_time = time.time() 
tmp_time = start_time

#1.이미지 파일 불러오기
from mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False,  one_hot_label=True) #(60000,784)(60000,10)(10000,784)(10000,10)

#2.클래스 불러오기
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)   #클래스객체생성

#3.하이퍼 파라미터 설정
iters_num = 2                   #반복횟수
train_size = x_train.shape[0]   #훈련데이터의 양 60000
batch_size = 10                 #미니배치 크기 100
learning_rate = 0.1             #학습률

#경과기록
train_loss_list = []  
train_acc_list = []
test_acc_list = []

#4.학습 시작
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #(100,)
    x_batch = x_train[batch_mask]   #(10000,784)
    t_batch = t_train[batch_mask]   #(10000,10)
    
    #기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    #매개변수 갱신
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
        
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    
    train_loss_list.append(loss) #리스트에 요소 추가
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print(str(i+1) + " loss : "+str(train_loss_list[i]/batch_size))
    print(str(i+1) +"train acc : " + str(train_acc))
    print(str(i+1) +"test acc : " + str(test_acc))
    
    #1회 학습에 걸린시간
    mid_time = time.time()
    one = mid_time - tmp_time
    print(str(i+1) + " time : " + str(int(one)) + "초")
    tmp_time = mid_time    

#5.총 걸린시간    
end_time = time.time()
print("Running time : " + str(int(end_time-start_time)) + "초")    
