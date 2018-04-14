import urllib.request
import time   
import numpy as np
import pickle
from collections import OrderedDict
import os
from optimizer import *
from LayerNet import TwoLayerNet

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