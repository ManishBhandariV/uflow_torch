import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt



# params = torch.arange(50, dtype= torch.float32).reshape(1,5,5,2)
# indices = torch.tensor([[1,2,3,4],[5,6,7,8]])
#
# params_shape = torch.tensor(params.shape, dtype= torch.int32)
# indices_shape = torch.tensor(indices.shape, dtype= torch.int32)
#
# slice_dimensions = indices_shape[-1]
#
# max_index = params_shape[:slice_dimensions] - 1
# min_index = torch.zeros_like(max_index, dtype=torch.int32)
# # print(min_index)
# clipped_indices = torch.min(torch.max(indices, min_index), max_index)
# # clipped_indices = torch.clamp(indices, min_index, max_index)
# # print(clipped_indices)
# # print(min_index)
#
# x = torch.tensor([[True,  True], [False, False]])
# x = torch.prod(x, 1, dtype= params.dtype)
# #
# print(x)
#
# # print(torch.prod(x) > 0) # False
# # print(torch.prod(x, dim= 0) > 0)  # [False, False]
# print(torch.prod(x, 1))  # [True, False]

t5 = np.reshape(np.arange(18), [2, 3, 3])
t5 = torch.tensor(t5)


def gather_nd(params, indices):
  ind_shape = list(indices.shape)

  if len(ind_shape) ==2:

    slices = [t4[i[0]] for i in indices]
    return torch.stack(slices)

  if ind_shape == 3:




t4 = torch.arange(60).reshape(5,2,2,3)

indices = torch.tensor([[2], [3], [0]])

b = [t4[i[0]] for i in indices]

c = torch.stack(b)

print(c)
