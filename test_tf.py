import collections

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential

class PWCFlow(Model):
  """Model for estimating flow based on the feature pyramids of two images."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               dropout_rate=0.25,
               num_channels_upsampled_context=32,
               num_levels=5,
               normalize_before_cost_volume=True,
               channel_multiplier=1.,
               use_cost_volume=True,
               use_feature_warp=True,
               accumulate_flow=True,
               use_bfloat16=False,
               shared_flow_decoder=False):

    super(PWCFlow, self).__init__()
    self._use_bfloat16 = use_bfloat16
    if use_bfloat16:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_bfloat16')
    else:
      self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
          'float32')
    self._leaky_relu_alpha = leaky_relu_alpha
    self._drop_out_rate = dropout_rate
    self._num_context_up_channels = num_channels_upsampled_context
    self._num_levels = num_levels
    self._normalize_before_cost_volume = normalize_before_cost_volume
    self._channel_multiplier = channel_multiplier
    self._use_cost_volume = use_cost_volume
    self._use_feature_warp = use_feature_warp
    self._accumulate_flow = accumulate_flow
    self._shared_flow_decoder = shared_flow_decoder

    self._refine_model = self._build_refinement_model()
    self._flow_layers = self._build_flow_layers()
    if not self._use_cost_volume:
      self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs(
      )
    if num_channels_upsampled_context:
      self._context_up_layers = self._build_upsample_layers(
          num_channels=int(num_channels_upsampled_context * channel_multiplier))
    if self._shared_flow_decoder:
      # pylint:disable=invalid-name
      self._1x1_shared_decoder = self._build_1x1_shared_decoder()

  def call(self, feature_pyramid1, feature_pyramid2, training=False):
    """Run the model."""
    context = None
    flow = None
    flow_up = None
    context_up = None
    flows = []

    # Go top down through the levels to the second to last one to estimate flow.
    for level, (features1, features2) in reversed(
        list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

      # init flows with zeros for coarsest level if needed
      if self._shared_flow_decoder and flow_up is None:
        batch_size, height, width, _ = features1.shape.as_list()
        flow_up = tf.zeros(
            [batch_size, height, width, 2],
            dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)
        if self._num_context_up_channels:
          num_channels = int(self._num_context_up_channels *
                             self._channel_multiplier)
          context_up = tf.zeros(
              [batch_size, height, width, num_channels],
              dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

      # Warp features2 with upsampled flow from higher level.
      if flow_up is None or not self._use_feature_warp:
        warped2 = features2
      else:
        warp_up = uflow_utils.flow_to_warp(flow_up)
        warped2 = uflow_utils.resample(features2, warp_up)

      # Compute cost volume by comparing features1 and warped features2.
      features1_normalized, warped2_normalized = normalize_features(
          [features1, warped2],
          normalize=self._normalize_before_cost_volume,
          center=self._normalize_before_cost_volume,
          moments_across_channels=True,
          moments_across_images=True)

      if self._use_cost_volume:
        cost_volume = compute_cost_volume(
            features1_normalized, warped2_normalized, max_displacement=4)
      else:
        concat_features = Concatenate(axis=-1)(
            [features1_normalized, warped2_normalized])
        cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

      cost_volume = LeakyReLU(
          alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)(
              cost_volume)

      if self._shared_flow_decoder:
        # This will ensure to work for arbitrary feature sizes per level.
        conv_1x1 = self._1x1_shared_decoder[level]
        features1 = conv_1x1(features1)

      # Compute context and flow from previous flow, cost volume, and features1.
      if flow_up is None:
        x_in = Concatenate(axis=-1)([cost_volume, features1])
      else:
        if context_up is None:
          x_in = Concatenate(axis=-1)([flow_up, cost_volume, features1])
        else:
          x_in = Concatenate(axis=-1)(
              [context_up, flow_up, cost_volume, features1])

      # Use dense-net connections.
      x_out = None
      if self._shared_flow_decoder:
        # reuse the same flow decoder on all levels
        flow_layers = self._flow_layers
      else:
        flow_layers = self._flow_layers[level]
      for layer in flow_layers[:-1]:
        x_out = layer(x_in)
        x_in = Concatenate(axis=-1)([x_in, x_out])
      context = x_out

      flow = flow_layers[-1](context)

      if (training and self._drop_out_rate):
        maybe_dropout = tf.cast(
            tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
            tf.bfloat16 if self._use_bfloat16 else tf.float32)
        context *= maybe_dropout
        flow *= maybe_dropout

      if flow_up is not None and self._accumulate_flow:
        flow += flow_up

      # Upsample flow for the next lower level.
      flow_up = uflow_utils.upsample(flow, is_flow=True)
      if self._num_context_up_channels:
        context_up = self._context_up_layers[level](context)

      # Append results to list.
      flows.insert(0, flow)

    # Refine flow at level 1.
    refinement = self._refine_model([context, flow])
    if (training and self._drop_out_rate):
      refinement *= tf.cast(
          tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
          tf.bfloat16 if self._use_bfloat16 else tf.float32)
    refined_flow = flow + refinement
    flows[0] = refined_flow
    return [tf.cast(flow, tf.float32) for flow in flows]

  def _build_cost_volume_surrogate_convs(self):
    layers = []
    for _ in range(self._num_levels):
      layers.append(
          Conv2D(
              int(64 * self._channel_multiplier),
              kernel_size=(4, 4),
              padding='same',
              dtype=self._dtype_policy))
    return layers

  def _build_upsample_layers(self, num_channels):
    """Build layers for upsampling via deconvolution."""
    layers = []
    for unused_level in range(self._num_levels):
      layers.append(
          Conv2DTranspose(
              num_channels,
              kernel_size=(4, 4),
              strides=2,
              padding='same',
              dtype=self._dtype_policy))
    return layers

  def _build_flow_layers(self):
    """Build layers for flow estimation."""
    # Empty list of layers level 0 because flow is only estimated at levels > 0.
    result = [[]]
    for _ in range(1, self._num_levels):
      layers = []
      for c in [128, 128, 96, 64, 32]:
        layers.append(
            Sequential([
                Conv2D(
                    int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same',
                    dtype=self._dtype_policy),
                LeakyReLU(
                    alpha=self._leaky_relu_alpha, dtype=self._dtype_policy)
            ]))
      layers.append(
          Conv2D(
              2,
              kernel_size=(3, 3),
              strides=1,
              padding='same',
              dtype=self._dtype_policy))
      if self._shared_flow_decoder:
        return layers
      result.append(layers)
    return result

  def _build_refinement_model(self):
    """Build model for flow refinement using dilated convolutions."""
    layers = []
    layers.append(Concatenate(axis=-1))
    for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
      layers.append(
          Conv2D(
              int(c * self._channel_multiplier),
              kernel_size=(3, 3),
              strides=1,
              padding='same',
              dilation_rate=d,
              dtype=self._dtype_policy))
      layers.append(
          LeakyReLU(alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
    layers.append(
        Conv2D(
            2,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            dtype=self._dtype_policy))
    return Sequential(layers)

  def _build_1x1_shared_decoder(self):
    """Build layers for flow estimation."""
    # Empty list of layers level 0 because flow is only estimated at levels > 0.
    result = [[]]
    for _ in range(1, self._num_levels):
      result.append(
          Conv2D(
              32,
              kernel_size=(1, 1),
              strides=1,
              padding='same',
              dtype=self._dtype_policy))
    return result


model = PWCFlow()

tf.keras.utils.plot_model(model, show_shapes=True, dpi=48)
