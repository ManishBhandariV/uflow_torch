import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# import tensorflow as tf

a = list(range(61))
t = []
i = 0
while i != len(a)-1:
    if i!= 0 and  i%20 == 0:
        i=i+1
    t.append([a[i],a[i+1]])
    i+=1
print(t)

