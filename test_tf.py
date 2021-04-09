import tensorflow as tf
import numpy as np

# flow = tf.ones((1,2,2,2))
flow = np.arange(0,150).reshape((3,5,5,2))
flow = tf.Variable(flow, dtype= tf.float32)
# flow = tf.Variable([[[[1., 2.], [3., 4.], [5., 6.]],
#                       [[7., 8.], [9., 10.], [11., 12.]]]])
# print(flow.shape)
print(flow[:,0,:,:])
flow = tf.pad(tensor= flow , paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],mode='SYMMETRIC' )
# print(flow[:,0,:,:])

