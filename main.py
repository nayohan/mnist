#To run sub_directory python file on cloud IDE
import os, sys

#Folder List
value = ["00_neural_network","01_numerical_gradient","02_backpropagation","03_optimizer","04_cnn"] #

#Run Python File
os.system("python /projects/mnist_cnn/" + value[4] + "/main.py")      


"""
http://blog.danggun.net/4064

sudo apt-get install python-pip python-dev

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

sudo pip3 install --upgrade $TF_BINARY_URL
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow')
sess = tf.Session()
print(sess.run(hello))
