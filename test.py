import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# import tensorflow as tf
from absl import flags
import uflow_flags
import test_tf
import os
import torch

image = torch.ones((1,3,5,5), dtype = torch.float32)
# image = torch.moveaxis(image, 1, -1)
rgb_weights = torch.tensor([2., 4., 6.])
intensities = image[:,0,Ellipsis] *2 + image[:,1,Ellipsis] *4 + image[:,2,Ellipsis]*6
print(intensities.shape)
