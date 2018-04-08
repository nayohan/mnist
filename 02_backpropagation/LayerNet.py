from functions import *
from collections import OrderedDict
from backpropagation import *
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):     #초기 파라미터값 설정
        self.params = {}
        """
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  #(784,50)
        self.params['b1'] = np.zeros(hidden_size)                                       #(50,)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)  #(50,10)
        self.params['b2'] = np.zeros(output_size)                                       #(10,)
                
        #가중치값 쓰기
        f = open("/projects/mnist_cnn/02_backpropagation/y_w1.txt", 'w')
        f.write(str(self.params['W1']))
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_b1.txt", 'w')
        f.write(str(self.params['b1']))
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_w2.txt", 'w')
        f.write(str(self.params['W2']))
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_b2.txt", 'w')
        f.write(str(self.params['b2']))
        f.close()
        """
        https://wikidocs.net/33#pickle
      
           
        #가중치값 불러오기기
        f = open("/projects/mnist_cnn/02_backpropagation/y_w1.txt", 'r')
        self.params['W1'] = np.asarray(f.readlines())
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_b1.txt", 'r')
        self.params['b1'] = np.asarray(f.read().split())
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_w2.txt", 'r')
        self.params['W2'] = np.asarray(f.read().split())
        f.close()
        f = open("/projects/mnist_cnn/02_backpropagation/y_b2.txt", 'r')
        self.params['b2'] = np.asarray(f.read().split())
        f.close()
        
        print(self.params['W1'].shape)
        print(self.params['W1'])
        print(self.params['b1'].shape)
        print(self.params['W2'].shape)
        print(self.params['b2'].shape)        
        #계층생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):       #(100,784)                                              #이미지 예측 
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