import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf

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

params = torch.ones((1,32,40,40), dtype= torch.float32)
indices = torch.arange(1,4801, dtype = torch.float32).reshape(1,3, 40,40)

params = torch.moveaxis(params,1,-1)
indices = torch.moveaxis(indices,1,-1)
params_shape = torch.tensor(params.shape, dtype= torch.float32)
indices_shape = torch.tensor(indices.shape, dtype= torch.float32)
slice_dimensions = indices_shape[-1]

max_index = params_shape[:slice_dimensions.type(torch.int)] - 1
min_index = torch.zeros_like(max_index, dtype=torch.int32)
clipped_indices = indices.detach().clone()

clipped_indices[:,:,:,0][clipped_indices[:,:,:,0] > max_index[0]] = max_index[0]
clipped_indices[:,:,:,1][clipped_indices[:,:,:,1] > max_index[1] ] = max_index[1]
clipped_indices[:,:,:,2][clipped_indices[:,:,:,2] > max_index[2] ] = max_index[2]

clipped_indices[clipped_indices < 0] = 0

mask = torch.prod(
    torch.logical_and(indices >= min_index, indices <= max_index), -1, dtype=params.dtype)
mask = torch.unsqueeze(mask, -1)
mask = mask.numpy()
# print(mask)
################tensorflow#######################

params_tf = tf.ones((1,40,40,32), dtype = tf.float32)
indices_tf = tf.reshape(tf.range(1,4801, dtype = tf.float32),(1,40,40,3))

params_shape_tf = tf.shape(params_tf)
indices_shape = tf.shape(indices_tf)
slice_dimensions_tf = indices_shape[-1]

max_index_tf = params_shape_tf[:slice_dimensions_tf] - 1
max_index_tf = tf.cast(max_index_tf,tf.float32)
min_index_tf = tf.zeros_like(max_index_tf, dtype=tf.float32)

clipped_indices_tf = tf.clip_by_value(indices_tf, min_index_tf, max_index_tf)

# Check whether each component of each index is in range [min, max], and
# allow an index only if all components are in range:
mask_tf = tf.reduce_all(
  tf.logical_and(indices_tf >= min_index_tf, indices_tf <= max_index_tf), -1)
mask_tf = tf.expand_dims(mask_tf, -1)
mask_tf = mask_tf.numpy()

# print(np.array_equal(mask_tf,mask))


# a = torch.ones((1,2,2,3))
# a = torch.tensor([[[[1., 2., 3.],
#                     [4., 5., 6.]],
#
#                     [[7., 8., 9.],
#                     [10., 11., 12.]]]])
# # print(a[a[:,:,:,0] !=1])
# # maski = a[:,:,:,0] >1
# a[:,:,:,0][(a[:,:,:,0]>1)]  = 0
# a = torch.arange(1,13, dtype= torch.float32).reshape(1,3, 2,2,)
# d = a.permute(0,2,3,1)
# d = d.permute(0,2,1,3)
# # c = torch.moveaxis(a,1,-1)
# # c = torch.moveaxis(c,1,2)
# # print(c.shape)
# # print(c)
# b = tf.reshape(tf.range(1,13, dtype= tf.float32),(1,2,2,3))
# c = tf.transpose(b, [0,3,1,2])
#
# print(b)
# print(d)
# print(c)

def resample(source, coords):
    """Resample the source image at the passed coordinates.
    Args:
      source: tf.tensor, batch of images to be resampled.
      coords: [B, 2, H, W] tf.tensor, batch of coordinates in the image.
    Returns:
      The resampled image.
    Coordinates should be between 0 and size-1. Coordinates outside of this range
    are handled by interpolating with a background image filled with zeros in the
    same way that SAME size convolution works.
    """

    _, _, h, w = source.shape
    # normalize coordinates to [-1 .. 1] range
    coords = coords.clone()
    coords[:, 0, :, :] = 2.0 * coords[:, 0, :, :].clone() / max(h - 1, 1) - 1.0
    coords[:, 1, :, :] = 2.0 * coords[:, 1, :, :].clone() / max(w - 1, 1) - 1.0
    coords = coords.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(source, coords, align_corners=False)
    return output