# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for resampling images."""

import torch
import uflow_gpu_utils


# def torch_gather_nd(params, indices):
#   params = torch.moveaxis(params, 1, -1)
#   indices = torch.moveaxis(indices, 1, -1)
#   if len(indices.shape) <= 2:
#     ans = [params[(*indi,)] for indi in indices]
#     return torch.stack(ans)
#   elif len(indices.shape) == 3:
#     b = []
#     for i in indices:
#       a = []
#       for j in i:
#         a.append(params[(*j,)])
#       b.append(torch.tensor(a))
#     return torch.stack(b)
#   else:
#     c = []
#     for i in indices:
#       b = []
#       for j in i:
#         a = []
#         for k in j:
#           a.append(params[(*k,)])
#         b.append(torch.stack(a))
#       c.append(torch.stack(b))
#     d = torch.stack(c)
#     d = torch.moveaxis(d, -1, 1)
#     return d

def gather_nd(params, indices):
  params = torch.moveaxis(params, (0, 1, 2, 3), (0, 3, 1, 2))
  indices = torch.moveaxis(indices, (0, 1, 2, 3), (0, 3, 1, 2))
  indices = indices.type(torch.int64)
  gathered = params[list(indices.T)]
  gathered = torch.moveaxis(gathered, (0, 1, 2, 3), (1, 2, 0, 3))
  gathered = torch.moveaxis(gathered, (0, 1, 2, 3), (0, 2, 3, 1))

  return gathered


def safe_gather_nd(params, indices):
  """Gather slices from params into a Tensor with shape specified by indices.

  Similar functionality to tf.gather_nd with difference: when index is out of
  bound, always return 0.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
      tensor.

  Returns:
    A Tensor. Has the same type as params. Values from params gathered from
    specified indices (if they exist) otherwise zeros, with shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:].
  """
  params_shape = torch.tensor(params.shape, dtype= torch.int32)
  max_index = params_shape - 1
  min_index = torch.zeros_like(max_index, dtype=torch.int32)

  # clipped_indices = torch.min(torch.max(indices, min_index), max_index)
  clipped_indices = torch.stack([indices[:, 0, :, :].clamp(0, max_index[0]), indices[:, 1, :, :].clamp(0, max_index[2]), indices[:, 2, :, :].clamp(0, max_index[3])],dim= 1)
  # Check whether each component of each index is in range [min, max], and
  # allow an index only if all components are in range:

  mask_l = indices >= 0
  mask_u = torch.stack([indices[:,0,:,:] <= max_index[0], indices[:,1,:,:] <= max_index[2], indices[:,2,:,:] <= max_index[3]],dim= 1)

  mask = torch.prod(
      torch.logical_and(mask_l, mask_u), 1, dtype= params.dtype)
  mask = torch.unsqueeze(mask, 1)

  return (mask *  gather_nd(params,clipped_indices))


def resampler(data, warp):
  """Resamples input data at user defined coordinates.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor shape `[batch_size, dim_0, ... , dim_n, 2]` containing the
      coordinates at which resampling will be performed.
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
  """
  # data = torch.tensor(data)
  # warp = torch.tensor(warp)
  warp_x, warp_y = torch.unbind(warp, dim= 1)

  return resampler_with_unstacked_warp(data, warp_x, warp_y)


def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y,
                                  safe=True,
                                  ):
  """Resamples input data at user defined coordinates.

  The resampler functions in the same way as `resampler` above, with the
  following differences:
  1. The warp coordinates for x and y are given as separate tensors.
  2. If warp_x and warp_y are known to be within their allowed bounds, (that is,
     0 <= warp_x <= width_of_data - 1, 0 <= warp_y <= height_of_data - 1) we
     can disable the `safe` flag.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
      coordinates at which resampling will be performed.
    warp_y: Tensor of the same shape as warp_x containing the y coordinates at
      which resampling will be performed.
    safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
      Disable only if you know they are within bounds, otherwise a runtime
      exception will be thrown.
    name: Optional name of the op.

  Returns:
     Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ValueError: If warp_x, warp_y and data have incompatible shapes.
  """


  # warp_x = torch.tensor(warp_x)
  # warp_y = torch.tensor(warp_y)
  # data = torch.tensor(data)
  if not warp_x.shape == warp_y.shape:
    raise ValueError(
        'warp_x and warp_y are of incompatible shapes: %s vs %s ' %
        (str(warp_x.shape), str(warp_y.shape)))
  warp_shape = torch.tensor(warp_x.shape).to(uflow_gpu_utils.device)

  if warp_x.shape[0] != data.shape[0]:
    raise ValueError(
        '\'warp_x\' and \'data\' must have compatible first '
        'dimension (batch size), but their shapes are %s and %s ' %
        (str(warp_x.shape[0]), str(data.shape[0])))
  # Compute the four points closest to warp with integer value.
  warp_floor_x = torch.floor(warp_x)
  warp_floor_y = torch.floor(warp_y)
  # Compute the weight for each point.
  right_warp_weight = warp_x - warp_floor_x
  down_warp_weight = warp_y - warp_floor_y

  warp_floor_x = warp_floor_x.type(torch.int32)
  warp_floor_y = warp_floor_y.type(torch.int32)
  warp_ceil_x = torch.ceil(warp_x).type(torch.int32)
  warp_ceil_y = torch.ceil(warp_y).type(torch.int32)

  left_warp_weight = torch.subtract(
      torch.tensor(1.0, dtype= right_warp_weight.dtype), right_warp_weight)
  up_warp_weight = torch.subtract(
      torch.tensor(1.0, dtype= down_warp_weight.dtype), down_warp_weight)

  # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
  # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
  # dimension being the batch index.

  # A shape like warp_shape but with all sizes except the first set to 1:
  # warp_batch_shape = torch.cat(
  #     [warp_shape[0:1], torch.ones_like(warp_shape[1:])], 0)

  warp_batch = torch.arange(warp_shape[0], dtype=torch.int32).reshape(warp_shape[0:1],*torch.ones_like(warp_shape[1:]).tolist()).to(uflow_gpu_utils.device)

  # Broadcast to match shape:

  warp_batch = warp_batch +  torch.zeros_like(warp_y, dtype=torch.int32)
  left_warp_weight = torch.unsqueeze(left_warp_weight, dim=1)
  down_warp_weight = torch.unsqueeze(down_warp_weight, dim=1)
  up_warp_weight = torch.unsqueeze(up_warp_weight, dim = 1)
  right_warp_weight = torch.unsqueeze(right_warp_weight, dim = 1)

  up_left_warp = torch.stack([warp_batch, warp_floor_y, warp_floor_x], dim=1)
  up_right_warp = torch.stack([warp_batch, warp_floor_y, warp_ceil_x], dim=1)
  down_left_warp = torch.stack([warp_batch, warp_ceil_y, warp_floor_x], dim =1)
  down_right_warp = torch.stack([warp_batch, warp_ceil_y, warp_ceil_x], dim=1)

  def gather(params, indices):
    return safe_gather_nd(params,indices)if safe else gather_nd(params,indices)

  # gather data then take weighted average to get resample result.

  result = (
      (gather(data, up_left_warp) * left_warp_weight +
       gather(data, up_right_warp) * right_warp_weight) * up_warp_weight +
      (gather(data, down_left_warp) * left_warp_weight +
       gather(data, down_right_warp) * right_warp_weight) *
      down_warp_weight)

  return result

# a = torch.arange(1*40*40*32, dtype = torch.float32).reshape(1,32,40,40)
# b = torch.arange(1*40*40*2, dtype = torch.float32).reshape(1,2,40,40)
#
# c = resampler(a,b)