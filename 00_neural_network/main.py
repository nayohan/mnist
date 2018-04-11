#기본베이스로 실행되야하는 것
import urllib.request
import sys,os
import pickle
#mnist.py파일 경로
mnist_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/dataset/mnist.py"
#sample_weight.pkl 파일 경로
weight_url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch/master/ch03/sample_weight.pkl"
#다운로드 파일은 html파일이 받아지지 않도록 주의!!

a = urllib.request.urlopen(mnist_url)	#변수a에 파일저장
b = urllib.request.urlopen(weight_url)	#변수b에 파일저장

os.chdir('/projects/mnist_cnn/00_neural_network')			#파일경로 /home/sorna로 변경
k = open("mnist.py","wb")		#mnist.py 파일 생성 (같은 위치에 저장)
k.write(a.read())				#파일 쓰기
k.close()						#파일 닫기

h = open("sample_weight.pkl","wb")		#sampe_weight.pkl 파일 생성 (같은 위치에 저장)
h.write(b.read())				#파일 쓰기
h.close()	
			
#시작
import numpy as np				#넘파이 라이브러리 임포트
from mnist import load_mnist	#mnist.py파일에서 load_mnist함수 임포트

def sigmoid(x):					#시그모이드 함수
  return 1 / (1 + np.exp(-x)) 

def softmax(a):					#소프트맥스 함수
  c = np.max(a)
  exp_a = np.exp(a - c)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

def get_data():					#load_mnist함수를 이용해서 훈련,시험데이터 저장
  (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
  return x_test, t_test
  
def init_network():				#sample_weight파일에서 학습된 가중치 파일 network변수로로딩
  with open("sample_weight.pkl",'rb') as sample_w:
    network = pickle.load(sample_w)
    print(network)
  return network



def predict(network, x):		#시험데이터와 가중치파일을 이용해서 정확도 예측(정규화)
  W1,W2,W3 = network['W1'], network['W2'],  network['W3']
  b1,b2,b3 = network['b1'], network['b2'],  network['b3']
  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)
  return y

x,t = get_data()				#시험데이터 10000개이미지와 10000개 정답을 x,t에 저장

network = init_network()		#변수network에 10000개 가중치숫자 저장

accuracy_cnt = 0				#정답의 갯수

for i in range(len(x)):			#x는 10000개 
  y = predict(network, x[i])	#10000개 이미지,10000개 정답 정확도 예측
  p = np.argmax(y)				#확률이 가장높은 원소의 인덱스를 변수p에 저장
  if p == t[i]:					#정답과 비교
    accuracy_cnt += 1			#맞으면 +1
    
import matplotlib.pyplot as plt
from matplotlib.image import imread

for i in [1,2,3]:
	img = x[i]
	img = img.reshape(28,28)
	plt.imshow(img)
	plt.show()
	print(t[i])
   

print("Accuracy:" + str(float(accuracy_cnt) /len(x)))	#정확도 출력 0.9207