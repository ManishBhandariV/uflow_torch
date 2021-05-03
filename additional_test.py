import torch
import tensorflow as tf

a = tf.ones((1,32,32,1))
b = a.shape.as_list()
print(b[-3:-1])