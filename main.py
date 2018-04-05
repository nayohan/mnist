#URL의 파일다운
import urllib.request
mnist_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/dataset/mnist.py"
a=urllib.request.urlopen(mnist_url)
k=open("mnist.py","wb")
k.write(a.read())
k.close()

#시간 확인 
import time   
start_time = time.time() 
tmp_time = start_time

#이미지 파일 불러오기
from mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False,  one_hot_label=True) #(60000,784)(60000,10)(10000,784)(10000,10)
train_loss_list = []  

#클래스 불러오기
from LayerNet import TwoLayerNet
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)   #클래스객체생성

#하이퍼 파라미터
iters_num = 2                   #반복횟수
train_size = x_train.shape[0]   #훈련데이터의 양 60000
batch_size = 10                 #미니배치 크기 100
learning_rate = 0.1             #학습률

#학습 시작
import numpy as np
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
    train_loss_list.append(loss) #리스트에 요소 추가
    print(str(i+1) + " loss : "+str(train_loss_list[i]/batch_size))
    
    #1회 학습에 걸린시간
    mid_time = time.time()
    one = mid_time - tmp_time
    print(str(i+1) + " time : " + str(int(one)) + "초")
    tmp_time = mid_time
    
#총 걸린시간    
end_time = time.time()
print("Running time : " + str(int(end_time-start_time)) + "초")    
