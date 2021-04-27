import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt

# params= torch.arange(1*32*80*80, dtype= torch.float32).reshape(1,32, 80,80)
# # indices = torch.arange(1*3*40,40, dtype= torch.float32).reshape(1,3,40,40)
# indices = torch.zeros((1,1,80,80),dtype= torch.long)



# params = torch.tensor(params)
# indices = torch.tensor( [[0,1],[1,0]])

def torch_gather_nd(params, indices):
    params = torch.moveaxis(params,1,-1)
    indices = torch.moveaxis(indices,1,-1)
    if len(indices.shape) <= 2:
        ans = [params[(*indi,)] for indi in indices]
        return torch.stack(ans)
    elif len(indices.shape) == 3:
        b = []
        for i in indices:
            a = []
            for j in i:
              a.append(params[(*j,)])
            b.append(torch.tensor(a))
        return torch.stack(b)
    else:
        c = []
        for i in indices:
            b = []
            for j in i:
                a = []
                for k in j:
                    a.append(params[(*k,)])
                b.append(torch.stack(a))
            c.append(torch.stack(b))
        d = torch.stack(c)
        d = torch.moveaxis(d,-1,1)
        return d

# print(torch_gather_nd(params,indices).shape)
# a = torch.arange(1,1*3*3*3+1,dtype = torch.float32).reshape(3,3)
#
# b = torch.nn.ConstantPad2d((1,1,1,1),0)
#
# c = b(a)
# print(c)
# print(a)
# print(c[1:-1,1:-1])
# a = [128, 128, 96, 64, 32]
# for i in range(len(a)):
#     if i== 0:
#         b = 128
#     else:  b = 113 + sum(a[:i])
#     print(b)
# params = torch.arange(1*40*40*32, dtype = torch.float32).reshape(1,32,40,40)
# indices = torch.arange(1*40*40*3, dtype = torch.float32).reshape(1,3,40,40)
# params_shape = torch.tensor(params.shape, dtype= torch.int32)
# max_index = params_shape - 1
# min_index = torch.zeros_like(max_index, dtype=torch.int32)
#
# c = torch.max(indices, min_index)
# clipped_indices = torch.min(torch.max(indices, min_index), max_index)
a = torch.arange(1,13, dtype = torch.float32).reshape(1,3,2,2)
mask1 = a[:,:,:,0] <= 1
mask2 = a[:,:,:,1] <= 3
mask3 = a[:,:,:,2] <= 2
mask4 = a[:,:,:,3] <=2

#this  will give me a mask along all dimeniosn. Combine thi smask so that it has the shape of th einput tensor and apply thi snask to the indices
