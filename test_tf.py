import tensorflow as tf
import numpy as np

# flow = tf.ones((1,2,2,2))

def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    flow: tf.tensor representing optical flow.

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """

  # Construct a grid of the image coordinates.
  height, width = flow.shape.as_list()[-3:-1]
  i_grid, j_grid = tf.meshgrid(
      tf.linspace(0.0, height - 1.0, int(height)),
      tf.linspace(0.0, width - 1.0, int(width)),
      indexing='ij')
  grid = tf.stack([i_grid, j_grid], axis=2)

  # Potentially add batch dimension to match the shape of flow.
  if len(flow.shape) == 4:
    grid = grid[None]

  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = tf.cast(grid, flow.dtype)
  warp = grid + flow
  return warp

def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """

  def _resize(img, mask=None):
    # _, orig_height, orig_width, _ = img.shape.as_list()
    orig_height = tf.shape(input=img)[1]
    orig_width = tf.shape(input=img)[2]

    if orig_height == height and orig_width == width:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = tf.math.multiply(img, mask)
      # resize image
      img_resized = tf.compat.v2.image.resize(
          img, (int(height), int(width)), antialias=True)
      # resize mask (will serve as normalization weights)
      mask_resized = tf.compat.v2.image.resize(
          mask, (int(height), int(width)), antialias=True)
      # normalize sparse flow field and mask
      img_resized = tf.math.multiply(img_resized,
                                     tf.math.reciprocal_no_nan(mask_resized))
      mask_resized = tf.math.multiply(mask_resized,
                                      tf.math.reciprocal_no_nan(mask_resized))
    else:
      # normal resize without anti-alaising
      img_resized = tf.compat.v2.image.resize(img, (int(height), int(width)))

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = tf.reshape([
          float(height) / tf.cast(orig_height, tf.float32),
          float(width) / tf.cast(orig_width, tf.float32)
      ], [1, 1, 1, 2])
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if len(shape) == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _resize(img, mask)
  elif len(shape) > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    img_flattened = tf.reshape(img, [-1] + shape[-3:])
    if mask is not None:
      mask_flattened = tf.reshape(mask, [-1] + shape[-3:])
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    # There appears to be some bug in tf2 tf.function
    # that fails to capture the value of height / width inside the closure,
    # leading the height / width undefined here. Call set_shape to make it
    # defined again.
    img_resized.set_shape(
        (img_resized.shape[0], height, width, img_resized.shape[3]))
    result_img = tf.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
    if mask is not None:
      mask_resized.set_shape(
          (mask_resized.shape[0], height, width, mask_resized.shape[3]))
      result_mask = tf.reshape(mask_resized,
                               shape[:-3] + mask_resized.shape[-3:])
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


def compute_range_map(flow,
                      downsampling_factor=1,
                      reduce_downsampling_bias=True,
                      resize_output=True):
  """Count how often each coordinate is sampled.

  Counts are assigned to the integer coordinates around the sampled coordinates
  using weights from bilinear interpolation.

  Args:
    flow: A float tensor of shape (batch size x height x width x 2) that
      represents a dense flow field.
    downsampling_factor: An integer, by which factor to downsample the output
      resolution relative to the input resolution. Downsampling increases the
      bin size but decreases the resolution of the output. The output is
      normalized such that zero flow input will produce a constant ones output.
    reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
      near the image boundaries by padding the flow field.
    resize_output: A boolean, whether to resize the output ot the input
      resolution.

  Returns:
    A float tensor of shape [batch_size, height, width, 1] that denotes how
    often each pixel is sampled.
  """

  # Get input shape.
  input_shape = flow.shape.as_list()
  if len(input_shape) != 4:
    raise NotImplementedError()
  batch_size, input_height, input_width, _ = input_shape

  flow_height = input_height
  flow_width = input_width

  # Apply downsampling (and move the coordinate frame appropriately).
  output_height = input_height // downsampling_factor
  output_width = input_width // downsampling_factor
  if downsampling_factor > 1:
    # Reduce the bias that comes from downsampling, where pixels at the edge
    # will get lower counts that pixels in the middle of the image, by padding
    # the flow field.
    if reduce_downsampling_bias:
      p = downsampling_factor // 2
      flow_height += 2 * p
      flow_width += 2 * p
      # Apply padding in multiple steps to padd with the values on the edge.
      for _ in range(p):
        flow = tf.pad(
            tensor=flow,
            paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC')
      coords = flow_to_warp(flow) - p
    # Update the coordinate frame to the downsampled one.
    coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
  elif downsampling_factor == 1:
    coords = flow_to_warp(flow)
  else:
    raise ValueError('downsampling_factor must be an integer >= 1.')

  # Split coordinates into an integer part and a float offset for interpolation.
  coords_floor = tf.floor(coords)
  coords_offset = coords - coords_floor
  coords_floor = tf.cast(coords_floor, 'int32')

  # Define a batch offset for flattened indexes into all pixels.
  batch_range = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  idx_batch_offset = tf.tile(
      batch_range, [1, flow_height, flow_width]) * output_height * output_width

  # Flatten everything.
  coords_floor_flattened = tf.reshape(coords_floor, [-1, 2])
  coords_offset_flattened = tf.reshape(coords_offset, [-1, 2])
  idx_batch_offset_flattened = tf.reshape(idx_batch_offset, [-1])

  # Initialize results.
  idxs_list = []
  weights_list = []

  # Loop over differences di and dj to the four neighboring pixels.
  for di in range(2):
    for dj in range(2):

      # Compute the neighboring pixel coordinates.
      idxs_i = coords_floor_flattened[:, 0] + di
      idxs_j = coords_floor_flattened[:, 1] + dj
      # Compute the flat index into all pixels.
      idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

      # Only count valid pixels.
      mask = tf.reshape(
          tf.compat.v1.where(
              tf.logical_and(
                  tf.logical_and(idxs_i >= 0, idxs_i < output_height),
                  tf.logical_and(idxs_j >= 0, idxs_j < output_width))), [-1])
      valid_idxs = tf.gather(idxs, mask)
      valid_offsets = tf.gather(coords_offset_flattened, mask)

      # Compute weights according to bilinear interpolation.
      weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
      weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
      weights = weights_i * weights_j

      # Append indices and weights to the corresponding list.
      idxs_list.append(valid_idxs)
      weights_list.append(weights)

  # Concatenate everything.
  idxs = tf.concat(idxs_list, axis=0)
  weights = tf.concat(weights_list, axis=0)

  # Sum up weights for each pixel and reshape the result.
  counts = tf.math.unsorted_segment_sum(
      weights, idxs, batch_size * output_height * output_width)
  count_image = tf.reshape(counts, [batch_size, output_height, output_width, 1])

  if downsampling_factor > 1:
    # Normalize the count image so that downsampling does not affect the counts.
    count_image /= downsampling_factor**2
    if resize_output:
      count_image = resize(
          count_image, input_height, input_width, is_flow=False)

  return count_image

img = tf.reshape(tf.range(0,77056),((1,1,172,224,2)))
shape = img.shape.as_list()
# flow = tf.Variable(flow, dtype= tf.float32)
# f = flow.shape.as_list()
# ans = compute_range_map(flow)
# print(f)
# height =600
# width = 600
# orig_height = 325
# orig_width = 1242

img_flattened = tf.reshape(img, [-1] + shape[-3:])
print(img_flattened.shape)
