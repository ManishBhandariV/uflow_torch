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
import random

num_sequence = 2000
data_tuples = []
sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [13, 14, 15, 16, 17, 18, 19, 20]]
for i in range(num_sequence):
    for js in sequences:
        image_files = ['%06d_%02d.png' %(i, j) for j in js]
        image_tuples = [[image_files[i], image_files[i+1]] for i in range(len(image_files)-1)]
        data_tuples.append(image_tuples)

random.shuffle(data_tuples)
data_tuples = [pair for seq in data_tuples for pair in seq]

print(data_tuples)






