#To run sub_directory python file on cloud IDE
import os, sys

#Folder List
value = ["00_neural_network","01_numerical_gradient","02_backpropagation","03_optimizer","04_cnn"] #

#Run Python File
os.system("python /projects/mnist_cnn/" + value[3] + "/main.py")      

