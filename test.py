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
# x = torch.tensor([[0.9752, 0.5587, 0.0972],
#         [0.9534, 0.2731, 0.6953]])
# l = torch.tensor([[0.2, 0.3, 0.]])
# u = torch.tensor([[0.8, 1., 0.65]])
# x = torch.arange(1*3*2*2, dtype = torch.float32).reshape(1,3,2,2)
x = tf.reshape(tf.range(12, dtype = tf.float32),(1,2,2,3))
x = torch.from_numpy(x.numpy())
x = torch.stack([x[Ellipsis,0],x[Ellipsis,1],x[Ellipsis,2]], dim= 1)
clamped = torch.stack([x[:,0,:,:].clamp(0,1), x[:,1,:,:].clamp(0,2), x[:,2,:,:].clamp(0,4)])

print(clamped)
# y = tf.clip_by_value(x,[0,0,0],[1,2,4])
# #
# # x = torch.where(x[0,:,:,:] > 1., 1., x)
# print(x[:,0])
#
# A = torch.rand(1,10,8,8)
# Means = torch.mean(torch.mean(A,dim=3),dim=2).unsqueeze(-1).unsqueeze(-1)
# Ones = torch.ones(A.size())
# Zeros = torch.zeros(A.size())
# Thresholded = torch.where(A > Means, Ones, Zeros)
