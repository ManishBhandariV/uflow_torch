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
    transform = transforms.ColorJitter(hue=[0, max_delta_hue])
    images = transform(images)

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
