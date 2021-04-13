import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # flow = torch.arange(0,77056,1,dtype= torch.float32).reshape((172,224,2))
# # flow
# # img = np.load("/home/manish/Desktop/000000_10.npy")
#
# # img = np.moveaxis(img,-1, 0)
# # print(img.shape)
#
# # print(flow.shape)
# # # flow = torch.ones((3,172,224,2))
# # # # flow = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
# # # #                       [[7., 8.], [9., 10.], [11., 12.]]]])
# # #
# #
# # # flow = flow[None]
# # # m = torch.nn.ReplicationPad3d((0,0,1,1,1,1))
# # # flow = m(flow)
# # # flow = torch.squeeze(flow,0)
# # # flow = F.pad(flow, pad= (0,0,1,1), mode= "replicate")
# # # print(flow[:,0,:,:])
# #
# # a = transforms.Compose([transforms.Resize((600,600))])
# # img = torch.Tensor(img[None])
# # img = torch.moveaxis(img,-1,1)
# #
# # img = a(img)
# # print(img.shape)
# # img = torch.moveaxis(img,1,-1)
# # plt.imshow(flow)
# # plt.show()
#
# # height =600
# # width = 600
# # orig_height = img.shape[1]
# #
# # orig_width = img.shape[2]
# #
# #
# # img = torch.arange(77056).reshape((1,1,172,224,2))
# # shape = list(img.shape)
# # img_flattened = torch.reshape(img, [-1] + shape[-3:])
# # img_flattened.set_shape([-1] + shape[-3:])
#
# def resize(img, height, width, is_flow, mask=None):
#   """Resize an image or flow field to a new resolution.
#
#   In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
#   performed to account for missing flow entries in the sparse flow field. The
#   weighting is based on the resized mask, which determines the 'amount of valid
#   flow vectors' that contributed to each individual resized flow vector. Hence,
#   multiplying by the reciprocal cancels out the effect of considering non valid
#   flow vectors.
#
#   Args:
#     img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
#     height: int, heigh of new resolution
#     width: int, width of new resolution
#     is_flow: bool, flag for scaling flow accordingly
#     mask: tf.tensor, mask (optional) per pixel {0,1} flag
#
#   Returns:
#     Resized and potentially scaled image or flow field (and mask).
#   """
#
#   def _resize(img, mask=None):
#     # _, orig_height, orig_width, _ = img.shape.as_list()
#     orig_height = img.shape[1]
#     orig_width = img.shape[2]
#
#     if orig_height == height and orig_width == width:
#       # early return if no resizing is required
#       if mask is not None:
#         return img, mask
#       else:
#         return img
#
#     if mask is not None:
#       # multiply with mask, to ensure non-valid locations are zero
#       img = img * mask
#       # resize image
#       resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
#       img = torch.moveaxis(img,-1,1)
#       img_resized = resize_transform(img)
#       img_resized = torch.moveaxis(img_resized,1,-1)
#       # resize mask (will serve as normalization weights)
#       mask = torch.moveaxis(mask,-1,1)
#       mask_resized = resize_transform(mask)
#       mask_resized = torch.moveaxis(mask_resized,1,-1)
#       mask_resized_reciprocal = torch.reciprocal(mask_resized)
#       mask_resized_reciprocal[mask_resized_reciprocal == float("inf")] = 0
#       # normalize sparse flow field and mask
#       img_resized = img_resized * mask_resized_reciprocal
#       mask_resized = mask_resized * mask_resized_reciprocal
#
#     else:
#       # normal resize without anti-alaising
#       resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
#       img = torch.moveaxis(img, -1, 1)
#       img_resized = resize_transform(img)
#       img_resized = torch.moveaxis(img_resized, 1, -1)
#     if is_flow:
#       # If image is a flow image, scale flow values to be consistent with the
#       # new image size.
#       scaling = torch.reshape(
#           torch.tensor([float(height) / orig_height,
#                         float(width) / orig_width])
#           , [1, 1, 1, 2])
#       img_resized *= scaling
#
#     if mask is not None:
#       return img_resized, mask_resized
#     return img_resized
#
#   # Apply resizing at the right shape.
#   shape = list(img.shape)
#   if len(shape) == 3:
#     if mask is not None:
#       img_resized, mask_resized = _resize(img[None], mask[None])
#       return img_resized[0], mask_resized[0]
#     else:
#       return _resize(img[None])[0]
#   elif len(shape) == 4:
#     # Input at the right shape.
#     return _resize(img, mask)
#   elif len(shape) > 4:
#     # Reshape input to [b, h, w, c], resize and reshape back.
#     img_flattened = torch.reshape(img, [-1] + shape[-3:])
#     if mask is not None:
#       mask_flattened = torch.reshape(mask, [-1] + shape[-3:])
#       img_resized, mask_resized = _resize(img_flattened, mask_flattened)
#     else:
#       img_resized = _resize(img_flattened)
#     # There appears to be some bug in tf2 tf.function
#     # that fails to capture the value of height / width inside the closure,
#     # leading the height / width undefined here. Call set_shape to make it
#     # defined again.
#     # img_resized.set_shape(
#     #     (img_resized.shape[0], height, width, img_resized.shape[3]))
#     result_img = torch.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
#     if mask is not None:
#       # mask_resized.set_shape(
#       #     (mask_resized.shape[0], height, width, mask_resized.shape[3]))
#       result_mask = torch.reshape(mask_resized,
#                                shape[:-3] + mask_resized.shape[-3:])
#       return result_img, result_mask
#     return result_img
#   else:
#     raise ValueError('Cannot resize an image of shape', shape)
#
# def flow_to_warp(flow):
#   """Compute the warp from the flow field.
#
#   Args:
#     flow: tensor representing optical flow.
#
#   Returns:
#     The warp, i.e. the endpoints of the estimated flow.
#   """
#
#   # Construct a grid of the image coordinates.
#   height, width = list(flow.shape)[-3:-1]
#   i_grid, j_grid = torch.meshgrid(
#       torch.linspace(0.0,height - 1.0, steps= int(height)),
#       torch.linspace(0.0,width - 1.0, steps= int(width)))
#
#   grid = torch.stack((i_grid,j_grid), dim= 2)
#
#   # Potentially add batch dimension to match the shape of flow.
#   if len(flow.shape) == 4:
#     grid = grid[None]
#
#   # Add the flow field to the image grid.
#   if flow.dtype != grid.dtype:
#     grid = grid.type(flow.dtype)
#   warp = grid + flow
#   return warp
#
# def compute_range_map(flow,
#                       downsampling_factor=1,
#                       reduce_downsampling_bias=True,
#                       resize_output=True):
#   """Count how often each coordinate is sampled.
#
#   Counts are assigned to the integer coordinates around the sampled coordinates
#   using weights from bilinear interpolation.
#
#   Args:
#     flow: A float tensor of shape (batch size x height x width x 2) that
#       represents a dense flow field.
#     downsampling_factor: An integer, by which factor to downsample the output
#       resolution relative to the input resolution. Downsampling increases the
#       bin size but decreases the resolution of the output. The output is
#       normalized such that zero flow input will produce a constant ones output.
#     reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
#       near the image boundaries by padding the flow field.
#     resize_output: A boolean, whether to resize the output to the input
#       resolution.
#
#   Returns:
#     A float tensor of shape [batch_size, height, width, 1] that denotes how
#     often each pixel is sampled.
#   """
#
#   # Get input shape.
#   input_shape = list(flow.shape)
#   if len(input_shape) != 4:
#     raise NotImplementedError()
#   batch_size, input_height, input_width, _ = input_shape
#
#   flow_height = input_height
#   flow_width = input_width
#
#   # Apply downsampling (and move the coordinate frame appropriately).
#   output_height = input_height // downsampling_factor
#   output_width = input_width // downsampling_factor
#   if downsampling_factor > 1:
#     # Reduce the bias that comes from downsampling, where pixels at the edge
#     # will get lower counts than pixels in the middle of the image, by padding
#     # the flow field.
#     if reduce_downsampling_bias:
#       p = downsampling_factor // 2
#       flow_height += 2 * p
#       flow_width += 2 * p
#       # Apply padding in multiple steps to padd with the values on the edge.
#       for _ in range(p):
#          flow = flow[None]
#          m = torch.nn.ReplicationPad3d((0, 0, 1, 1, 1, 1))
#          flow = m(flow)
#          flow = torch.squeeze(flow, 0)
#       coords = flow_to_warp(flow) - p
#     # Update the coordinate frame to the downsampled one.
#     coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
#   elif downsampling_factor == 1:
#     coords = flow_to_warp(flow)
#   else:
#     raise ValueError('downsampling_factor must be an integer >= 1.')
#
#   # Split coordinates into an integer part and a float offset for interpolation.
#   coords_floor = torch.floor(coords)
#   coords_offset = coords - coords_floor
#   coords_floor = coords.type(torch.int32)
#
#   # Define a batch offset for flattened indexes into all pixels.
#   batch_range = torch.reshape(torch.arange(batch_size), [batch_size, 1, 1])
#   idx_batch_offset = torch.tile(
#       batch_range, [1, flow_height, flow_width]) * output_height * output_width
#
#   # Flatten everything.
#   coords_floor_flattened = torch.reshape(coords_floor, [-1, 2])
#   coords_offset_flattened = torch.reshape(coords_offset, [-1, 2])
#   idx_batch_offset_flattened = torch.reshape(idx_batch_offset, [-1])
#
#   # Initialize results.
#   idxs_list = []
#   weights_list = []
#
#   # Loop over differences di and dj to the four neighboring pixels.
#   for di in range(2):
#     for dj in range(2):
#
#       # Compute the neighboring pixel coordinates.
#       idxs_i = coords_floor_flattened[:, 0] + di
#       idxs_j = coords_floor_flattened[:, 1] + dj
#       # Compute the flat index into all pixels.
#       idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j
#
#       # Only count valid pixels.
#       mask = torch.reshape(torch.stack(
#           torch.where(
#               torch.logical_and(
#                   torch.logical_and(idxs_i >= 0, idxs_i < output_height),
#                   torch.logical_and(idxs_j >= 0, idxs_j < output_width)))), [-1])
#       valid_idxs = torch.gather(idxs, 0, mask)
#
#
#       # valid_offsets = torch.gather(coords_offset_flattened, 0, mask)
#       valid_offsets = coords_offset_flattened[mask]
#       # Compute weights according to bilinear interpolation.
#       weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
#       weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
#       weights = weights_i * weights_j
#
#       # Append indices and weights to the corresponding list.
#       idxs_list.append(valid_idxs)
#       weights_list.append(weights)
#
#   # Concatenate everything.
#   idxs = torch.cat(idxs_list, dim=0)
#   weights = torch.cat(weights_list, dim=0)
#
#   # Sum up weights for each pixel and reshape the result.
#   counts = torch.zeros(batch_size * output_height * output_width).scatter_add(0, idxs, weights)
#   count_image = torch.reshape(counts, [batch_size, output_height, output_width, 1])
#
#   if downsampling_factor > 1:
#     # Normalize the count image so that downsampling does not affect the counts.
#     count_image /= downsampling_factor**2
#     if resize_output:
#       count_image = resize(
#           count_image, input_height, input_width, is_flow=False)
#
#   return count_image
#
# flow = torch.arange(231168, dtype= torch.float32).reshape((3,172,224,2))
# # print(flow.dtype)
# ans = compute_range_map(flow)
# print(ans.shape)
a = torch.arange(60, dtype = torch.float32).reshape((3,5,2,2))

b = image[offset_height:offset_height +target_height, offset_width: offset_width+ target_width, 0:num_channels ]
