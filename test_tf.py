import tensorflow as tf
import numpy as np


# image = tf.reshape(tf.range(346752, dtype= tf.float32),(3,172,224,3))
batch_size = 3
height = 172
width = 224
flow_01 = np.ones((batch_size, height, width, 2)) * 4.
flow_01 = tf.Variable(flow_01.astype(np.float32))
# flow_01 = tf.convert_to_tensor(value=flow_01.astype(np.float32))
print(flow_01.dtype)




