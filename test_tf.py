import tensorflow as tf

def random_crop(batch, max_offset_height=32, max_offset_width=32):
  """Randomly crop a batch of images.

  Args:
    batch: a 4-D tensor of shape [batch_size, height, width, num_channels].
    max_offset_height: an int, the maximum vertical coordinate of the top left
      corner of the cropped result.
    max_offset_width: an int, the maximum horizontal coordinate of the top left
      corner of the cropped result.

  Returns:
    a pair of 1) the cropped images in form of a tensor of shape
    [batch_size, height-max_offset, width-max_offset, num_channels],
    2) an offset tensor of shape [batch_size, 2] for height and width offsets.
  """

  # Compute current shapes and target shapes of the crop.
  batch_size, height, width, num_channels = batch.shape
  target_height = height - max_offset_height
  target_width = width - max_offset_width

  # Randomly sample offsets.
  offsets_height = (0 - (max_offset_height + 1)) * torch.rand([batch_size]) + (max_offset_height + 1)
  offsets_height = offsets_height.type(torch.int32)
  offsets_width = (0 - (max_offset_width + 1)) * torch.rand([batch_size]) + (max_offset_width + 1)
  offsets_width = offsets_width.type(torch.int32)

  offsets = torch.stack([offsets_height, offsets_width], dim=-1)

  # Loop over the batch and perform cropping.
  cropped_images = []
  for image, offset_height, offset_width in zip(batch, offsets_height,
                                                offsets_width):
    cropped_images.append(
        tf.slice(
            image,
            begin=[offset_height, offset_width, 0],
            size=[target_height, target_width, num_channels]))
  cropped_batch = torch.stack(cropped_images)

  return cropped_batch, offsets