import collections

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential

a = tf.reshape(tf.range(2*3*3, dtype= tf.float32),(3,3,2))
# # a = tf.Variable(a)
# b = tf.reshape(tf.zeros((1,40,40,1), dtype= tf.int32),(1,40,40,1))
# b = tf.Variable(b)
# b[0,39,39,:] = 0
warp = tf.reshape(tf.range(40*40, dtype=tf.int32),(1,40,40))
warp_shape = tf.shape(warp)

warp_batch_shape = tf.concat(
        [warp_shape[0:1], tf.ones_like(warp_shape[1:])], 0)
warp_batch = tf.reshape(tf.range(warp_shape[0], dtype=tf.int32),warp_batch_shape)
warp_batch += tf.zeros_like(warp, dtype= tf.int32)
print(tf.reduce_sum(warp_batch))