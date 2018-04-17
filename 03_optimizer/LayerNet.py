from functions import *
from collections import OrderedDict
from backpropagation import *
import numpy as np
import pickle

class TwoLayerNet:
  	#초기 파라미터값 설정
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):     
        self.params = {}
        if 0:
            #가중치 초기값 생성시 작동할부분
            """
            #가우시안 표준 정규 분포
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  
            self.params['b1'] = np.zeros(hidden_size)                                     
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
            self.params['b2'] = np.zeros(output_size)
                        
            #Xavier initialization
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) / np.sqrt(input_size) 
            self.params['b1'] = np.zeros(hidden_size)                                     
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size) / np.sqrt(input_size) 
            self.params['b2'] = np.zeros(output_size)
            """
            #He initialization
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) / np.sqrt(input_size/2) 
            self.params['b1'] = np.zeros(hidden_size)                                     
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size) / np.sqrt(input_size/2) 
            self.params['b2'] = np.zeros(output_size)
            

            #가중치 값 저장하기
            f = open("/projects/mnist_cnn/03_optimizer/weight_Adam.pkl", 'wb')
            pickle.dump(self.params, f)
            f.close()
           
  
        #가중치 값 불러오기기
        f = open("/projects/mnist_cnn/03_optimizer/weight_Adam.pkl", 'rb')
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