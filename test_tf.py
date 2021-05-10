import collections

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential

flow_gt = tf.ones((1,320,320,2))
occ_mask_gt = tf.ones_like(flow_gt[Ellipsis, -1:])
print(occ_mask_gt.shape)




