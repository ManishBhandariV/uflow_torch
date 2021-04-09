import tensorflow as tf

flow = tf.ones((1,3,4,2))
print(flow)
flow = tf.pad(tensor= flow , paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],mode='SYMMETRIC' )
print(flow)

