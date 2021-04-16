import tensorflow as tf
import numpy as np


params = tf.reshape(tf.range(50, dtype= tf.float32),(1,5,5,2))
indices = tf.Variable([1,2,3,4])

params_shape = tf.shape(params)
indices_shape = tf.shape(indices)
slice_dimensions = indices_shape[-1]

max_index = params_shape[:slice_dimensions] - 1
min_index = tf.zeros_like(max_index, dtype=tf.int32)

clipped_indices = tf.clip_by_value(indices, min_index, max_index)

# Check whether each component of each index is in range [min, max], and
# allow an index only if all components are in range:
mask = tf.reduce_all(
  tf.logical_and(indices >= min_index, indices <= max_index), -1)

print(mask)
# print(max_index)
# print(min_index)
mask = tf.expand_dims(mask, -1)




