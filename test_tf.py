import tensorflow as tf


image = tf.reshape(tf.range(346752, dtype= tf.float32),(3,172,224,3))

b = tf.nn.avg_pool(image, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
print(b.shape)




