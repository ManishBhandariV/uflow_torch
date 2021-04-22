import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt


a = torch.arange(5*172*224*3, dtype= torch.float32).reshape(5,3, 172, 224)
# a = F.pad(a,(1,1,1,1))

b =torch.nn.ConvTranspose2d(in_channels= 3, out_channels= 32,
                                      kernel_size= (4,4), stride= (2,2), padding=(1,1))

c = b(a)

print(c.shape)

