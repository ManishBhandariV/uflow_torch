import tensorflow as tf
import numpy as np
import torch

a = tf.reshape(tf.range(5*172*224*3, dtype= tf.float32),(5,172,224,3))

b = tf.keras.layers.Conv2DTranspose(
              32,
              kernel_size=(4, 4),
              strides=2,
              padding='same',
              )
print(b(a).shape)


