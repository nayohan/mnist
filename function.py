import numpy as np

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
    delta = 1e - 7 #0.0000001
    return -np.sum(t * np.log(y + delta))