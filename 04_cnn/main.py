import urllib.request
import os

mnist_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/dataset/mnist.py"
a=urllib.request.urlopen(mnist_url)
os.chdir('/projects/mnist_cnn/04_cnn')
k=open("mnist.py","wb")
k.write(a.read())
k.close()

import numpy as np
from mnist import load_mnist
from LayerNet import SimpleConvNet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
train_loss_list = []   

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
iters_num = 200                 #반복횟수
train_size = x_train.shape[0]   #훈련데이터의 양 60000
batch_size = 10                 #미니배치 크기 100
learning_rate = 0.1             #학습률

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size) #(100,)
    x_batch = x_train[batch_mask]   #(10000,784)
    t_batch = t_train[batch_mask]   #(10000,10)
    print(str(batch_mask.shape))
    #기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    #매개변수 갱신
    for key in ('W1','b1','W2','b2','W3','b3'):
        network.params[key] -= learning_rate * grad[key]
        
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) #리스트에 요소 추가
    print(str(i+1) + " loss : "+str(train_loss_list[i]/batch_size))
    
