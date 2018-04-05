from functions import *

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