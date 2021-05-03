import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# import tensorflow as tf

def _avg_pool3x3(x):
  return torch.nn.functional.avg_pool2d(x, (3,3), (1,1))

def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
  """Computes a weighted structured image similarity measure.

  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.

  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.

  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
  if c1 == float('inf') and c2 == float('inf'):
    raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                     'likely unintended.')
  weight = torch.unsqueeze(weight, 1)
  average_pooled_weight = _avg_pool3x3(weight)
  weight_plus_epsilon = weight + weight_epsilon
  inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

  def weighted_avg_pool3x3(z):
    wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
    return wighted_avg * inverse_average_pooled_weight

  mu_x = weighted_avg_pool3x3(x)
  mu_y = weighted_avg_pool3x3(y)
  sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
  sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
  sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
  if c1 == float('inf'):
    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
  elif c2 == float('inf'):
    ssim_n = 2 * mu_x * mu_y + c1
    ssim_d = mu_x**2 + mu_y**2 + c1
  else:
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  result = ssim_n / ssim_d
  return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


image1 = torch.arange(1*160*160*32, dtype= torch.float32).reshape(1,32, 160,160)
image2 = np.random.randint(low= 0, high = 256, size=(1,32, 160,160))
image2 =torch.tensor(image2, dtype= torch.float32)

mask = np.random.randint(low= 0, high= 1, size=(1,160,160))
mask = torch.tensor(mask, dtype = torch.float32)

a = weighted_ssim(image1,image2, mask)
print(a[0].shape, a[1].shape)

