from functools import partial
from math import pi
import torch
from torchvision import  transforms
import uflow_utils

def apply_augmentation(images, flow=None, mask=None,
                       crop_height=640, crop_width=640):
  """Applies photometric and geometric augmentations to images and flow."""
  # ensure sequence length of two, to be able to unstack images

  assert images.shape[0] == 2
  # apply geometric augmentation functions
  images, flow, mask = geometric_augmentation(
      images, flow, mask, crop_height, crop_width)
  # apply photometric augmentation functions
  images_aug = photometric_augmentation(images)

  # return flow and mask if available
  if flow is not None:
    return images_aug, images, flow, mask
  return images_aug, images

def photometric_augmentation(images,
                             augment_color_swap=True,
                             augment_hue_shift=True,
                             augment_saturation=False,
                             augment_brightness=False,
                             augment_contrast=False,
                             augment_gaussian_noise=False,
                             augment_brightness_individual=False,
                             augment_contrast_individual=False,
                             max_delta_hue=0.5,
                             min_bound_saturation=0.8,
                             max_bound_saturation=1.2,
                             max_delta_brightness=0.1,
                             min_bound_contrast=0.8,
                             max_bound_contrast=1.2,
                             min_bound_gaussian_noise=0.0,
                             max_bound_gaussian_noise=0.02,
                             max_delta_brightness_individual=0.02,
                             min_bound_contrast_individual=0.95,
                             max_bound_contrast_individual=1.05):
  """Applies photometric augmentations to an image pair."""
  # Randomly permute colors by rolling and reversing.
  # This covers all permutations.
  if augment_color_swap:
    r = torch.randint(size = [], high= 3,dtype =torch.int32)

    images = torch.roll(images, shifts=int(r.numpy()), dims = 1)


  if augment_hue_shift:
    images = transforms.ColorJitter(hue=[0, max_delta_hue])(images)

  if augment_saturation:
    images = transforms.ColorJitter(saturation= [min_bound_saturation, max_bound_saturation])(images)

  if augment_brightness:
    images = transforms.ColorJitter(brightness= [0, max_delta_brightness])(images)

  if augment_contrast:
    images = transforms.ColorJitter(contrast=[min_bound_contrast, max_bound_contrast])(images)



  image_1, image_2 = torch.unbind(images,0)

  image_1 = torch.clamp(image_1, 0.0, 1.0)
  image_2 = torch.clamp(image_2, 0.0, 1.0)

  return torch.stack([image_1, image_2])

def geometric_augmentation(images,
                           flow=None,
                           mask=None,
                           crop_height=640,
                           crop_width=640,
                           augment_flip_left_right=False,
                           augment_flip_up_down=False,
                           augment_scale=False,
                           augment_relative_scale=False,
                           augment_rotation=False,
                           augment_relative_rotation=False,
                           augment_crop_offset=False,
                           min_bound_scale=0.9,
                           max_bound_scale=1.5,
                           min_bound_relative_scale=0.95,
                           max_bound_relative_scale=1.05,
                           max_rotation_deg=15,
                           max_relative_rotation_deg=3,
                           max_relative_crop_offset=5):

  return images, flow, mask


def build_selfsup_transformations(num_flow_levels=3,
                                  seq_len=2,
                                  crop_height=0,
                                  crop_width=0,
                                  max_shift_height=0,
                                  max_shift_width=0,
                                  resize=True):
  """Apply augmentations to a list of student images."""

  def transform(images, i_or_ij, is_flow, crop_height, crop_width,
                shift_heights, shift_widths, resize):
    # Expect (i, j) for flows and masks and i for images.
    if isinstance(i_or_ij, int):
      i = i_or_ij
      # Flow needs i and j.
      assert not is_flow
    else:
      i, j = i_or_ij

    if is_flow:
      shifts = torch.stack([shift_heights, shift_widths], dim=-1)
      flow_offset = shifts[i] - shifts[j]
      images = images + flow_offset.type(torch.float32)

    shift_height = shift_heights[i]
    shift_width = shift_widths[i]
    height = images.shape[-2]
    width = images.shape[-1]

    # Assert that the cropped bounding box does not go out of the image frame.

    a = crop_height + shift_height >= 0
    b = crop_width + shift_width >= 0
    c = height - crop_height + shift_height <= height
    d = width - crop_width + shift_width <= width
    e = height > 2 * crop_height
    if not e:
      print('Image height is too small for cropping.')
    f = width > 2 * crop_width
    if not f:
      print('Image height is too small for cropping.')


    if a and b and c and d and e and f:
      images = images[:, :, crop_height + shift_height:height - crop_height +
                      shift_height, crop_width + shift_width:width -
                      crop_width + shift_width]
    if resize:
      images = uflow_utils.resize(images, height, width, is_flow=is_flow)

    return images

  max_divisor = 2**(num_flow_levels - 1)
  assert crop_height % max_divisor == 0
  assert crop_width % max_divisor == 0
  assert max_shift_height <= crop_height
  assert max_shift_width <= crop_width
  # Compute random shifts for different images in a sequence.
  if max_shift_height > 0 or max_shift_width > 0:
    max_rand = max_shift_height // max_divisor
    shift_height_at_highest_level = torch.randint(size=[seq_len],low= max_rand, high=max_rand + 1, dtype=torch.int32)
    shift_heights = shift_height_at_highest_level * max_divisor

    max_rand = max_shift_height // max_divisor
    shift_width_at_highest_level = torch.randint(size=[seq_len], low=max_rand, high=max_rand + 1, dtype=torch.int32)
    shift_widths = shift_width_at_highest_level * max_divisor

  transform_fns = []
  for level in range(num_flow_levels):

    if max_shift_height == 0 and max_shift_width == 0:
      shift_heights = [0, 0]
      shift_widths = [0, 0]
    else:
      shift_heights = shift_heights // (2**level)
      shift_widths = shift_widths // (2**level)

    fn = partial(
        transform,
        crop_height=crop_height // (2**level),
        crop_width=crop_width // (2**level),
        shift_heights=shift_heights,
        shift_widths=shift_widths,
        resize=resize)
    transform_fns.append(fn)
  assert len(transform_fns) == num_flow_levels
  return transform_fns
