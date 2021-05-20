import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import tensorflow as tf
from absl import flags
import uflow_flags
import test_tf
import os
import torch
import random
from PIL import Image
b = torch.square(torch.rand([1])).numpy()
a = tf.square(tf.random.uniform([1])).numpy()
print(a)
print(b)


