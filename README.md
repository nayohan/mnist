# MNIST_CNN


This project refer to book 'Deep Learning from scratch'. Machine recognize digit using machine learning with mnist database. 
The book explain python basics, perceptron, artificial neural network, convoulution neural network and deep learning.

I am student learning about machine learning from september 24, 2017. Uloading code on github from April 5,2018 until learn TensorFlow. It 
work on making convolution neural network. 

<hr/>

# Explain 
## 00_neural_network
 Make neural network using python. I wrote down the details of the functions below.
 
 * [main.py]()
   * [sigmoid(x)](https://github.com/nayohan/mnist_cnn/wiki#32-activation-function)
   * [softmax(a)]()
   * [get_data()]()
   * [init_network()]()
   * [predict(network, x)]()
   
## 01_numerical_gradient
To training handwriting digit data. Use numerical gradient. It is easy to implement. but it takse a long time for execute. It takes 1 iters per 2minutes. When we training MNIST data, we need 100batchsize*10000iters. If you calculate, Almost 14days need.

 * [LayerNet.py]()
   * [__init__(self, input_size, hidden_size, output_size, weight_init_std=0.01)]()
   * [predict(self, x)]()
   * [loss(self,x,t)]()
   * [accuracy(self,x,t)]()
   * [gradient(self,x,t)]()
   
 * [functions.py]()
   * [sigmoid(x)]()
   * [softmax(a)]()
   * [cross_entropy_error(y,t)]()
   * [numerical_gradient(f, x)]()
   
 * [main.py]()
   * []()
 
## 02_backpropagation
Now we need to reduce the time using another algorithm. Replace numerical gradient with backpropagation. It is hard to understand backpropagation. but using calculate graph, it can be easily understand. If we use backpropagation, exectuion time is highly decreased. 1iters per 1seconds. 120x faster. It takes 16minutes for training 1000batchsize*60iters*16ephocs.(Run on Codenvy.com)

 * [backpropagation.py]()
    * [Relu]()
    * [Sigmoid]() 
    * [Affine]()
    * [SoftmaxWithLoss]()
    
## 03_optimizer
The above is good enough. but the sooner the better. Using another parameter update algorithm, weight initial value, batch normalization and dropout. Actually in my code didn't applicable all thing. It's need to be modified. Additionally this version has matplotlib.pyplot.
That can show graph after the execution.

 * [optimizer.py]()
    * [SGD]()
    * [Momentum]()
    * [Nesterov]()
    * [AdaGrad]()
    * [RMSprop]()
    * [Adam]()
    
## 04_cnn
Convolution neural network is different upper code. IT
 * [backpropagation.py]()
    * [im2col]()
    * [col2im]() 
    * [Convolution]()
    * [Pooling]()

 * [Trainer.py]()
    
<hr/>

# Trouble shooting
