import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
# import tensorflow as tf

patch_size = 3
image = torch.arange(346752, dtype= torch.float32).reshape(3,172,224,3)

b = torch.moveaxis(torch.nn.functional.avg_pool2d(torch.moveaxis(image,-1,1), (3,3), (1,1)),1,-1)

print(b.shape)