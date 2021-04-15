import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
# import tensorflow as tf


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
    orig_height = img.shape[1]
    orig_width = img.shape[2]

    if orig_height == height and orig_width == width:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = img * mask
      # resize image
      resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
      img = torch.moveaxis(img,-1,1)
      img_resized = resize_transform(img)
      img_resized = torch.moveaxis(img_resized,1,-1)
      # resize mask (will serve as normalization weights)
      mask = torch.moveaxis(mask,-1,1)
      mask_resized = resize_transform(mask)
      mask_resized = torch.moveaxis(mask_resized,1,-1)
      mask_resized_reciprocal = torch.reciprocal(mask_resized)
      mask_resized_reciprocal[mask_resized_reciprocal == float("inf")] = 0
      # normalize sparse flow field and mask
      img_resized = img_resized * mask_resized_reciprocal
      mask_resized = mask_resized * mask_resized_reciprocal

    else:
      # normal resize without anti-alaising
      resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
      img = torch.moveaxis(img, -1, 1)
      img_resized = resize_transform(img)
      img_resized = torch.moveaxis(img_resized, 1, -1)
    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = torch.reshape(
          torch.tensor([float(height) / orig_height,
                        float(width) / orig_width])
          , [1, 1, 1, 2])
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = list(img.shape)
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
    img_flattened = torch.reshape(img, [-1] + shape[-3:])
    if mask is not None:
      mask_flattened = torch.reshape(mask, [-1] + shape[-3:])
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    # There appears to be some bug in tf2 tf.function
    # that fails to capture the value of height / width inside the closure,
    # leading the height / width undefined here. Call set_shape to make it
    # defined again.
    # img_resized.set_shape(
    #     (img_resized.shape[0], height, width, img_resized.shape[3]))
    result_img = torch.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
    if mask is not None:
      # mask_resized.set_shape(
      #     (mask_resized.shape[0], height, width, mask_resized.shape[3]))
      result_mask = torch.reshape(mask_resized,
                               shape[:-3] + mask_resized.shape[-3:])
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


# a = torch.arange(231168, dtype= torch.float32).reshape(3,172,224,2)
#
# b = resize(a,600,600,is_flow= True)

flow = torch.tensor(
        [[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.float32)

flow_result = torch.tensor([[[0.25, 0], [0, 0]], [[0, 0], [0, 0]]],
                              dtype=torch.float32)


# print(flow_result.shape)
flow_resized = resize(
        flow, 2, 2, is_flow=True)

print(flow_resized)