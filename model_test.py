import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential

class PWCFeaturePyramid(Model):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=16,
               level1_num_1x1=0,
               original_layer_sizes=False,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half',
               use_bfloat16=False):
    """Constructor.

    Args:
      leaky_relu_alpha: Float. Alpha for leaky ReLU.
      filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
        of form (num_convs_per_group, num_filters_per_conv).
      level1_num_layers: How many layers and filters to use on the first
        pyramid. Only relevant if filters is None and original_layer_sizes
        is False.
      level1_num_filters: int, how many filters to include on pyramid layer 1.
        Only relevant if filters is None and original_layer_sizes if False.
      level1_num_1x1: How many 1x1 convolutions to use on the first pyramid
        level.
      original_layer_sizes: bool, if True, use the original PWC net number
        of layers and filters.
      num_levels: int, How many feature pyramid levels to construct.
      channel_multiplier: float, used to scale up or down the amount of
        computation by increasing or decreasing the number of channels
        by this factor.
      pyramid_resolution: str, specifies the resolution of the lowest (closest
        to input pyramid resolution)
      use_bfloat16: bool, whether or not to run in bfloat16 mode.
    """

    super(PWCFeaturePyramid, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'float32')
    self._channel_multiplier = channel_multiplier
    if num_levels > 6:
      raise NotImplementedError('Max number of pyramid levels is 6')
    if filters is None:
      if original_layer_sizes:
        # Orig - last layer
        filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                   (3, 196))[:num_levels]
      else:
        filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                   (3, 32), (3, 32), (3, 32))[:num_levels]
    assert filters
    assert all(len(t) == 2 for t in filters)
    assert all(t[0] > 0 for t in filters)

    self._leaky_relu_alpha = leaky_relu_alpha
    self._convs = []
    self._level1_num_1x1 = level1_num_1x1

    for level, (num_layers, num_filters) in enumerate(filters):
      group = []
      for i in range(num_layers):
        stride = 1
        if i == 0 or (i == 1 and level == 0 and
                      pyramid_resolution == 'quarter'):
          stride = 2
        conv = Conv2D(
            int(num_filters * self._channel_multiplier),
            kernel_size=(3,
                         3) if level > 0 or i < num_layers - level1_num_1x1 else
            (1, 1),
            strides=stride,
            padding='valid',
            dtype=self._dtype_policy)
        group.append(conv)
      self._convs.append(group)

  def call(self, x, split_features_by_sample=False):
    if self._use_bfloat16:
      x = tf.cast(x, tf.bfloat16)
    x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
    features = []
    for level, conv_tuple in enumerate(self._convs):
      for i, conv in enumerate(conv_tuple):
        if level > 0 or i < len(conv_tuple) - self._level1_num_1x1:
          x = tf.pad(
              tensor=x,
              paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
              mode='CONSTANT')
        x = conv(x)
        x = LeakyReLU(alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)(x)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].

      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]  # pylint: disable=g-complex-comprehension

    return features

a = tf.reshape(tf.range())
fmodel = PWCFeaturePyramid()