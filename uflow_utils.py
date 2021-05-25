import torch
from torchvision import transforms
import  torch.nn.functional as F
from collections import  defaultdict
import numpy as np
import time
import  uflow_plotting

import uflow_resampler
import uflow_gpu_utils

# from uflow import uflow_plotting
# from uflow.uflow_resampler import resampler


def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    flow: tensor representing optical flow.

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """

  # Construct a grid of the image coordinates.
  _, _, height, width = list(flow.shape)
  # height, width = list(flow.shape)[-3:-1]
  i_grid, j_grid = torch.meshgrid(
      torch.linspace(0.0,height - 1.0, steps= int(height)),
      torch.linspace(0.0,width - 1.0, steps= int(width)))

  grid = torch.stack((i_grid,j_grid), dim= 0).to(uflow_gpu_utils.device)

  # Potentially add batch dimension to match the shape of flow.
  if len(flow.shape) == 4:
    grid = grid[None]

  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = grid.type(flow.dtype)
  warp = grid + flow
  return warp


def mask_invalid(coords):
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.

  Returns:
    The mask showing which coordinates are valid.
  """
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  max_height = float(coords.shape[-2] - 1)
  max_width = float(coords.shape[-1] - 1)
  mask = torch.logical_and(
      torch.logical_and(coords[:, 0, :, :] >= 0.0,
                     coords[:, 0, :, :] <= max_height),
      torch.logical_and(coords[:, 1, :, :] >= 0.0,
                     coords[:, 1, :, :] <= max_width))

  mask = torch.unsqueeze(mask,1).type(torch.float32)

  a = torch.sum(mask)
  return mask

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

# def resample(source, coords):
#   """Resample the source image at the passed coordinates.
#
#   Args:
#     source: tensor, batch of images to be resampled.
#     coords: tensor, batch of coordinates in the image.
#
#   Returns:
#     The resampled image.
#
#   Coordinates should be between 0 and size-1. Coordinates outside of this range
#   are handled by interpolating with a background image filled with zeros in the
#   same way that SAME size convolution works.
#   """
#
#   # Wrap this function because it uses a different order of height/width dims.
#   orig_source_dtype = source.dtype
#   if source.dtype != torch.float32:
#     source = source.type(torch.float32)
#   if coords.dtype != torch.float32:
#     coords = coords.type(torch.float32)
#   coords_rank = len(coords.shape)
#   if coords_rank == 4:
#     output = uflow_resampler.resampler(source, coords)
#     # output = uflow_resampler.resampler(source, coords[:, ::-1, :, :])
#     if orig_source_dtype != source.dtype:
#       return output.type(orig_source_dtype)
#     return output
#   else:
#     raise NotImplementedError()


def compute_range_map(flow,
                      downsampling_factor=1,
                      reduce_downsampling_bias=True,
                      resize_output=True):
  """Count how often each coordinate is sampled.

  Counts are assigned to the integer coordinates around the sampled coordinates
  using weights from bilinear interpolation.

  Args:
    flow: A float tensor of shape (batch size x height x width x 2) that
      represents a dense flow field.
    downsampling_factor: An integer, by which factor to downsample the output
      resolution relative to the input resolution. Downsampling increases the
      bin size but decreases the resolution of the output. The output is
      normalized such that zero flow input will produce a constant ones output.
    reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
      near the image boundaries by padding the flow field.
    resize_output: A boolean, whether to resize the output to the input
      resolution.

  Returns:
    A float tensor of shape [batch_size, height, width, 1] that denotes how
    often each pixel is sampled.
  """

  # Get input shape.
  input_shape = list(flow.shape)
  if len(input_shape) != 4:
    raise NotImplementedError()
  batch_size, _,  input_height, input_width = input_shape

  flow_height = input_height
  flow_width = input_width

  # Apply downsampling (and move the coordinate frame appropriately).
  output_height = input_height // downsampling_factor
  output_width = input_width // downsampling_factor
  if downsampling_factor > 1:
    # Reduce the bias that comes from downsampling, where pixels at the edge
    # will get lower counts than pixels in the middle of the image, by padding
    # the flow field.
    if reduce_downsampling_bias:
      p = downsampling_factor // 2
      flow_height += 2 * p
      flow_width += 2 * p
      # Apply padding in multiple steps to padd with the values on the edge.
      for _ in range(p):
         # flow = flow[None]
         m = torch.nn.ReplicationPad2d((1, 1, 1, 1))
         flow = m(flow)
         # flow = torch.squeeze(flow, 0)
      coords = flow_to_warp(flow) - p
    # Update the coordinate frame to the downsampled one.
    coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
  elif downsampling_factor == 1:
    coords = flow_to_warp(flow)
  else:
    raise ValueError('downsampling_factor must be an integer >= 1.')

  # Split coordinates into an integer part and a float offset for interpolation.
  coords_floor = torch.floor(coords)
  coords_offset = coords - coords_floor
  coords_floor = coords.type(torch.int32)

  # Define a batch offset for flattened indexes into all pixels.
  batch_range = torch.reshape(torch.arange(batch_size), [batch_size, 1, 1])
  idx_batch_offset = torch.tile(
      batch_range, [1, flow_height, flow_width]) * output_height * output_width
  idx_batch_offset = idx_batch_offset.to(uflow_gpu_utils.device)
  # Flatten everything.

  coords_floor_flattened = torch.reshape(torch.squeeze(coords_floor,0), [2, -1])
  coords_offset_flattened = torch.reshape(torch.squeeze(coords_offset,0), [2, -1])
  idx_batch_offset_flattened = torch.reshape(idx_batch_offset, [-1])

  # Initialize results.
  idxs_list = []
  weights_list = []

  # Loop over differences di and dj to the four neighboring pixels.
  for di in range(2):
    for dj in range(2):

      # Compute the neighboring pixel coordinates.
      idxs_i = coords_floor_flattened[0, :] + di
      idxs_j = coords_floor_flattened[1, :] + dj
      # Compute the flat index into all pixels.
      idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

      # Only count valid pixels.
      mask = torch.reshape(torch.stack(
          torch.where(
              torch.logical_and(
                  torch.logical_and(idxs_i >= 0, idxs_i < output_height),
                  torch.logical_and(idxs_j >= 0, idxs_j < output_width)))), [-1])
      valid_idxs = torch.gather(idxs, 0, mask)
      valid_offsets = torch.gather(coords_offset_flattened, 1, torch.stack((mask,mask), dim = 0))

      # Compute weights according to bilinear interpolation.
      weights_i = (1. - di) - (-1)**di * valid_offsets[0]
      weights_j = (1. - dj) - (-1)**dj * valid_offsets[1]
      weights = weights_i * weights_j

      # Append indices and weights to the corresponding list.
      idxs_list.append(valid_idxs)
      weights_list.append(weights)

  # Concatenate everything.
  idxs = torch.cat(idxs_list, dim=0)
  weights = torch.cat(weights_list, dim=0)

  # Sum up weights for each pixel and reshape the result.
  counts = torch.zeros(batch_size * output_height * output_width).to(uflow_gpu_utils.device).scatter_add(0, idxs, weights)
  count_image = torch.reshape(counts, [batch_size, 1, output_height, output_width])

  if downsampling_factor > 1:
    # Normalize the count image so that downsampling does not affect the counts.
    count_image /= downsampling_factor**2
    if resize_output:
      count_image = resize(
          count_image, input_height, input_width, is_flow=False)

  return count_image


def compute_warps_and_occlusion(flows,
                                occlusion_estimation,
                                occ_weights=None,
                                occ_thresholds=None,
                                occ_clip_max=None,
                                occlusions_are_zeros=True,
                                occ_active=None):
  """Compute warps, valid warp masks, advection maps, and occlusion masks."""




  if occ_clip_max is not None:
    for key in occ_clip_max:
      if key not in ['forward_collision', 'fb_abs']:
        raise ValueError('occ_clip_max for this key is not supported')

  warps = dict()
  range_maps_high_res = dict()
  range_maps_low_res = dict()
  occlusion_logits = dict()
  occlusion_scores = dict()
  occlusion_masks = dict()
  valid_warp_masks = dict()
  fb_sq_diff = dict()
  fb_sum_sq = dict()

  for key in flows:

    i, j, t = key
    rev_key = (j, i, t)

    warps[key] = []
    range_maps_high_res[key] = []
    range_maps_low_res[rev_key] = []
    occlusion_masks[key] = []
    valid_warp_masks[key] = []
    fb_sq_diff[key] = []
    fb_sum_sq[key] = []

    for level in range(min(3, len(flows[key]))):

      flow_ij = flows[key][level]
      flow_ji = flows[rev_key][level]


      # Compute warps (coordinates) and a mask for which coordinates are valid.

      warps[key].append(flow_to_warp(flow_ij))
      valid_warp_masks[key].append(mask_invalid(warps[key][level]))


      # Compare forward and backward flow.
      flow_ji_in_i = resample(flow_ji, warps[key][level])
      fb_sq_diff[key].append(
          ((flow_ij + flow_ji_in_i)**2).sum(dim=1, keepdim=True))
      fb_sum_sq[key].append(
          (flow_ij**2 + flow_ji_in_i**2).sum(dim=1, keepdim=True))

      if level != 0:
        continue

      # This initializations avoids problems in tensorflow (likely AutoGraph)
      occlusion_mask = torch.zeros_like(flow_ij[:, :1, Ellipsis], dtype=torch.float32)
      occlusion_scores['forward_collision'] = torch.zeros_like(
          flow_ij[:, :1, Ellipsis], dtype=torch.float32)
      occlusion_scores['backward_zero'] = torch.zeros_like(
          flow_ij[:, :1, Ellipsis], dtype= torch.float32)
      occlusion_scores['fb_abs'] = torch.zeros_like(
          flow_ij[:, :1, Ellipsis], dtype=torch.float32)

      if occlusion_estimation == 'none' or (
          occ_active is not None and not occ_active[occlusion_estimation]):
        occlusion_mask = torch.zeros_like(flow_ij[:, :1, Ellipsis], dtype=torch.float32)

      elif occlusion_estimation == 'brox':
        occlusion_mask = fb_sq_diff[key][level] > 0.01 * fb_sum_sq[key][level] + 0.5
        occlusion_mask = occlusion_mask.type(torch.float32)

      elif occlusion_estimation == 'fb_abs':
        occlusion_mask = fb_sq_diff[key][level]**0.5 > 1.5
        occlusion_mask = occlusion_mask.type(torch.float32)
      elif occlusion_estimation == 'wang':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=1,
                reduce_downsampling_bias=False,
                resize_output=False))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = (
            1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

      elif occlusion_estimation == 'wang4':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=4,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = (
            1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

      elif occlusion_estimation == 'wangthres':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=1,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = range_maps_low_res[rev_key][level] < 0.75,
        occlusion_mask = occlusion_mask.type(torch.float32)

      elif occlusion_estimation == 'wang4thres':
        range_maps_low_res[rev_key].append(
            compute_range_map(
                flow_ji,
                downsampling_factor=4,
                reduce_downsampling_bias=True,
                resize_output=True))
        # Invert so that low values correspond to probable occlusions,
        # range [0, 1].
        occlusion_mask = range_maps_low_res[rev_key][level] < 0.75
        occlusion_mask = occlusion_mask.type(torch.float32)

      elif occlusion_estimation == 'uflow':
        # Compute occlusion from the range map of the forward flow, projected
        # back into the frame of image i. The idea is if many flow vectors point
        # to the same pixel, those are likely occluded.
        if 'forward_collision' in occ_weights and (
            occ_active is None or occ_active['forward_collision']):
          range_maps_high_res[key].append(
              compute_range_map(
                  flow_ij,
                  downsampling_factor=1,
                  reduce_downsampling_bias=True,
                  resize_output=True))
          fwd_range_map_in_i = resample(range_maps_high_res[key][level],
                                        warps[key][level])
          # Rescale to [0, max-1].
          occlusion_scores['forward_collision'] = torch.clamp(
              fwd_range_map_in_i, 1., occ_clip_max['forward_collision']) - 1.0

        # Compute occlusion from the range map of the backward flow, which is
        # already computed in frame i. Pixels that no flow vector points to are
        # likely occluded.
        if 'backward_zero' in occ_weights and (occ_active is None or
                                               occ_active['backward_zero']):
          range_maps_low_res[rev_key].append(
              compute_range_map(
                  flow_ji,
                  downsampling_factor=4,
                  reduce_downsampling_bias=True,
                  resize_output=True))
          # Invert so that low values correspond to probable occlusions,
          # range [0, 1].
          occlusion_scores['backward_zero'] = (
              1. - torch.clamp(range_maps_low_res[rev_key][level], 0., 1.))

        # Compute occlusion from forward-backward consistency. If the flow
        # vectors are inconsistent, this means that they are either wrong or
        # occluded.
        if 'fb_abs' in occ_weights and (occ_active is None or
                                        occ_active['fb_abs']):
          # Clip to [0, max].
          occlusion_scores['fb_abs'] = torch.clamp(
              fb_sq_diff[key][level]**0.5, 0.0, occ_clip_max['fb_abs'])

        occlusion_logits = torch.zeros_like(flow_ij[:,:1,Ellipsis], dtype=torch.float32)
        for k, v in occlusion_scores.items():
          occlusion_logits += (v - occ_thresholds[k]) * occ_weights[k]
        occlusion_mask = torch.sigmoid(occlusion_logits)
      else:
        raise ValueError('Unknown value for occlusion_estimation:',
                         occlusion_estimation)

      occlusion_masks[key].append(
          1. - occlusion_mask if occlusions_are_zeros else occlusion_mask)

  return warps, valid_warp_masks, range_maps_low_res, occlusion_masks, fb_sq_diff, fb_sum_sq


def apply_warps_stop_grad(sources, warps, level):
  """Apply all warps on the correct sources."""

  warped = dict()
  for (i, j, t) in warps:
    # Only propagate gradient through the warp, not through the source.
    sources[j].requires_grad = False

    warped[(i, j, t)] = resample(
        sources[j], warps[(i, j, t)][level])

  return warped


def upsample(img, is_flow):
  """Double resolution of an image or flow field.

  Args:
    img: tf.tensor, image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """
  # _, height, width, _ = list(img.shape)
  _, _, height, width = list(img.shape)
  orig_dtype = img.dtype
  if orig_dtype != torch.float32:
    img = img.type(torch.float32)
  resize_transform = transforms.Compose([transforms.Resize((int(height*2), int(width*2)))])
  # img = torch.moveaxis(img, -1, 1)
  img_resized = resize_transform(img)
  # img_resized = torch.moveaxis(img_resized, 1, -1)

  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized *= 2
  if img_resized.dtype != orig_dtype:
    return img_resized.type(orig_dtype)
  return img_resized


def downsample(img, is_flow):
  """Halve the resolution of an image or flow field.

  Args:
    img: tf.tensor, image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """
  _, height, width, _ = list(img.shape)
  resize_transform = transforms.Compose([transforms.Resize((int(height/2), int(width/2)))])
  img = torch.moveaxis(img, -1, 1)
  img_resized = resize_transform(img)
  img_resized = torch.moveaxis(img_resized, 1, -1)

  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized /= 2
  return img_resized


def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """

  def _resize(img, mask=None):
    # _, orig_height, orig_width, _ = img.shape.as_list()
    orig_height = img.shape[2]
    orig_width = img.shape[3]

    if orig_height == height and orig_width == width:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = img * mask
      # resize image
      resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
      # img = torch.moveaxis(img,-1,1)
      img_resized = resize_transform(img)
      # img_resized = torch.moveaxis(img_resized,1,-1)
      # resize mask (will serve as normalization weights)
      # mask = torch.moveaxis(mask,-1,1)
      mask_resized = resize_transform(mask)
      # mask_resized = torch.moveaxis(mask_resized,1,-1)
      mask_resized_reciprocal = torch.reciprocal(mask_resized)
      mask_resized_reciprocal[mask_resized_reciprocal == float("inf")] = 0
      # normalize sparse flow field and mask
      img_resized = img_resized * mask_resized_reciprocal
      mask_resized = mask_resized * mask_resized_reciprocal

    else:
      # normal resize without anti-alaising
      resize_transform = transforms.Compose([transforms.Resize((int(height), int(width)))])
      # img = torch.moveaxis(img, -1, 1)
      img_resized = resize_transform(img)
      # img_resized = torch.moveaxis(img_resized, 1, -1)
    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = torch.reshape(
          torch.tensor([float(height) / orig_height,
                        float(width) / orig_width])
          , [1, 2, 1, 1]).to(uflow_gpu_utils.device)
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = list(img.shape)
  if len(shape) == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _resize(img, mask)
  elif len(shape) > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    img_flattened = torch.reshape(img, [-1] + shape[-3:])
    if mask is not None:
      mask_flattened = torch.reshape(mask, [-1] + shape[-3:])
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    # There appears to be some bug in tf2 tf.function
    # that fails to capture the value of height / width inside the closure,
    # leading the height / width undefined here. Call set_shape to make it
    # defined again.
    # img_resized.set_shape(
    #     (img_resized.shape[0], height, width, img_resized.shape[3]))
    result_img = torch.reshape(img_resized, shape[:-3] + list(img_resized.shape[-3:]))
    if mask is not None:
      # mask_resized.set_shape(
      #     (mask_resized.shape[0], height, width, mask_resized.shape[3]))
      result_mask = torch.reshape(mask_resized,
                               shape[:-3] + list(mask_resized.shape)[-3:])
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


def random_subseq(sequence, subseq_len):
  """Select a random subsequence of a given length."""
  seq_len = sequence.shape[0]
  start_index = torch.randint(size= [], low= 0, high= seq_len - subseq_len + 1)
  subseq = sequence[start_index:start_index + subseq_len]

  return subseq


def normalize_for_feature_metric_loss(features):
  """Normalize features for the feature-metric loss."""
  normalized_features = dict()
  for key, feature_map in features.items():
    # Normalize feature channels to have the same absolute activations.
    norm_feature_map = feature_map / (
            (abs(feature_map)).sum(dim=[0, 1, 2], keepdim=True) +
        1e-16)
    # Normalize every pixel feature across all channels to have mean 1.
    norm_feature_map /= (
        ((abs(norm_feature_map)).sum(dim=[-1], keepdim=True) +
        1e-16))
    normalized_features[key] = norm_feature_map
  return normalized_features


def l1(x):
  return torch.abs(x + 1e-6)


def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return torch.pow((torch.abs(diff) + eps), q)


def image_grads(image_batch, stride=1):
  image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
  image_batch_gw = image_batch[Ellipsis, stride:] - image_batch[Ellipsis, :-stride]
  return image_batch_gh, image_batch_gw


def image_averages(image_batch):
  image_batch_ah = (image_batch[:, 1:] + image_batch[:, :-1]) / 2.
  image_batch_aw = (image_batch[:, :, 1:] + image_batch[:, :, :-1]) / 2
  return image_batch_ah, image_batch_aw


def get_distance_metric_fns(distance_metrics):
  """Returns a dictionary of distance metrics."""
  output = {}
  for key, distance_metric in distance_metrics.items():
    if distance_metric == 'l1':
      output[key] = l1
    elif distance_metric == 'robust_l1':
      output[key] = robust_l1
    elif distance_metric == 'ddflow':
      output[key] = abs_robust_loss
    else:
      raise ValueError('Unknown loss function')
  return output


def compute_loss(
    weights,
    images,
    flows,
    warps,
    valid_warp_masks,
    not_occluded_masks,
    fb_sq_diff,
    fb_sum_sq,
    warped_images,
    only_forward=False,
    selfsup_transform_fns=None,
    fb_sigma_teacher=0.003,
    fb_sigma_student=0.03,
    plot_dir=None,
    distance_metrics=None,
    smoothness_edge_weighting='gaussian',
    stop_gradient_mask=True,
    selfsup_mask='gaussian',
    ground_truth_occlusions=None,
    smoothness_at_level=2,
):
  """Compute UFlow losses."""
  if distance_metrics is None:
    distance_metrics = {
        'photo': 'robust_l1',
        'census': 'ddflow',
    }
  distance_metric_fns = get_distance_metric_fns(distance_metrics)
  losses = dict()
  for key in weights:
    if key not in ['edge_constant']:
      losses[key] = 0.0

  compute_loss_for_these_flows = ['augmented-student']
  # Count number of non self-sup pairs, for which we will apply the losses.
  num_pairs = sum(
      [1.0 for (i, j, c) in warps if c in compute_loss_for_these_flows])

  # Iterate over image pairs.
  for key in warps:
    i, j, c = key

    if c not in compute_loss_for_these_flows or (only_forward and i > j):
      continue

    if ground_truth_occlusions is None:
      if stop_gradient_mask:
        mask_level0 = not_occluded_masks[key][0] * valid_warp_masks[key][0]
        # mask_level0.requires_grad = False
        mask_level0.detach()

      else:
        mask_level0 = not_occluded_masks[key][0] * valid_warp_masks[key][0]
    else:
      # For using ground truth mask
      if i > j:
        continue
      ground_truth_occlusions = ground_truth_occlusions.type(torch.float32)
      ground_truth_occlusions = 1.0 - ground_truth_occlusions

      mask_level0 = ground_truth_occlusions * valid_warp_masks[key][0]

      mask_level0.requires_grad = False
      height, width = list(valid_warp_masks[key][1].shape)[-2:]

    if 'photo' in weights:
      error = distance_metric_fns['photo'](images[i] - warped_images[key])
      losses['photo'] += (
          weights['photo'] * (mask_level0 * error).sum() /
          (mask_level0.sum() + 1e-16) / num_pairs)

      losses['photo'].requires_grad = True

    if 'smooth2' in weights or 'smooth1' in weights:

      edge_constant = 0.0
      if 'edge_constant' in weights:
        edge_constant = weights['edge_constant']

      abs_fn = None
      if smoothness_edge_weighting == 'gaussian':
        abs_fn = lambda x: x**2
      elif smoothness_edge_weighting == 'exponential':
        abs_fn = abs

      # Compute image gradients and sum them up to match the receptive field
      # of the flow gradients, which are computed at 1/4 resolution.
      images_level0 = images[i]
      height, width = list(images_level0.shape)[-2:]
      # Resize two times for a smoother result.
      images_level1 = resize(
          images_level0, int(height) // 2, int(width) // 2, is_flow=False)
      images_level2 = resize(
          images_level1, int(height) // 4, int(width) // 4, is_flow=False)
      images_at_level = [images_level0, images_level1, images_level2]

      if 'smooth1' in weights:

        img_gx, img_gy = image_grads(images_at_level[smoothness_at_level])
        weights_x = torch.exp(-((abs_fn(edge_constant * img_gx)).mean(dim=1, keepdim=True)))
        weights_y = torch.exp(-((abs_fn(edge_constant * img_gy)).mean(dim=1, keepdim=True)))

        # Compute second derivatives of the predicted smoothness.
        flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])

        # Compute weighted smoothness
        losses['smooth1'] += (
            weights['smooth1'] *
            ((weights_x * robust_l1(flow_gx)).sum() +
             (weights_y * robust_l1(flow_gy)).sum()) / 2. /
            num_pairs)
        losses['smooth1'].requires_grad = True

        if plot_dir is not None:
          uflow_plotting.plot_smoothness(key, images, weights_x, weights_y,
                                         robust_l1(flow_gx), robust_l1(flow_gy),
                                         flows, plot_dir)

      if 'smooth2' in weights:

        img_gx, img_gy = image_grads(
            images_at_level[smoothness_at_level], stride=2)
        weights_xx = torch.exp(-(
            (abs_fn(edge_constant * img_gx)).sum(dim=1, keepdim=True)))
        weights_yy = torch.exp(-((abs_fn(edge_constant * img_gy)).mean(dim=1, keepdim=True)))

        # Compute second derivatives of the predicted smoothness.
        flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])
        flow_gxx, unused_flow_gxy = image_grads(flow_gx)
        unused_flow_gyx, flow_gyy = image_grads(flow_gy)

        # Compute weighted smoothness
        losses['smooth2'] += (
            weights['smooth2'] *
            ((weights_xx * robust_l1(flow_gxx)).mean() +
             (weights_yy * robust_l1(flow_gyy)).mean()) /
            2. / num_pairs)
        # losses['smooth2'].requires_grad = True

        if plot_dir is not None:
          uflow_plotting.plot_smoothness(key, images, weights_xx, weights_yy,
                                         robust_l1(flow_gxx),
                                         robust_l1(flow_gyy), flows, plot_dir)

    if 'ssim' in weights:
      ssim_error, avg_weight = weighted_ssim(warped_images[key], images[i],
                                             torch.squeeze(mask_level0, dim= 1))

      losses['ssim'] += weights['ssim'] * (
          (ssim_error * avg_weight).sum() /
          ((avg_weight).sum() + 1e-16) / num_pairs)
      losses['ssim'].requires_grad = True

    if 'census' in weights:
      losses['census'] += weights['census'] * census_loss(
          images[i],
          warped_images[key],
          mask_level0,
          distance_metric_fn=distance_metric_fns['census']) / num_pairs
      # losses['census'].requires_grad = True
      print(losses['census'])

    if 'selfsup' in weights:
      assert selfsup_transform_fns is not None
      _, _, h, w = list(flows[key][2].shape)
      teacher_flow = flows[(i, j, 'original-teacher')][2]
      student_flow = flows[(i, j, 'transformed-student')][2]
      teacher_flow = selfsup_transform_fns[2](
          teacher_flow, i_or_ij=(i, j), is_flow=True)
      if selfsup_mask == 'gaussian':
        student_fb_consistency = torch.exp(
            -fb_sq_diff[(i, j, 'transformed-student')][2] /
            (fb_sigma_student**2 * (h**2 + w**2)))
        teacher_fb_consistency = torch.exp(
            -fb_sq_diff[(i, j, 'original-teacher')][2] / (fb_sigma_teacher**2 *
                                                          (h**2 + w**2)))
      elif selfsup_mask == 'advection':
        student_fb_consistency = not_occluded_masks[(i, j,
                                                     'transformed-student')][2]
        teacher_fb_consistency = not_occluded_masks[(i, j,
                                                     'original-teacher')][2]
      elif selfsup_mask == 'ddflow':
        threshold_student = 0.01 * (fb_sum_sq[
            (i, j, 'transformed-student')][2]) + 0.5
        threshold_teacher = 0.01 * (fb_sum_sq[
            (i, j, 'original-teacher')][2]) + 0.5
        student_fb_consistency = fb_sq_diff[(i, j, 'transformed-student')][2] < threshold_student
        student_fb_consistency = student_fb_consistency.type(torch.float32)
        teacher_fb_consistency = fb_sq_diff[(i, j, 'original-teacher')][2] < threshold_teacher
        teacher_fb_consistency = teacher_fb_consistency.type(torch.float32)
      else:
        raise ValueError('Unknown selfsup_mask', selfsup_mask)

      student_mask = 1. - (
          student_fb_consistency *
          valid_warp_masks[(i, j, 'transformed-student')][2])
      teacher_mask = (
          teacher_fb_consistency *
          valid_warp_masks[(i, j, 'original-teacher')][2])
      teacher_mask = selfsup_transform_fns[2](
          teacher_mask, i_or_ij=(i, j), is_flow=False)
      teacher_flow.requires_grad = False
      error = robust_l1(teacher_flow - student_flow)
      mask = teacher_mask * student_mask
      mask.requires_grad = False
      losses['selfsup'] += (
          weights['selfsup'] * (mask * error).sum() /
          ((torch.ones_like(mask)).sum() + 1e-16) / num_pairs)

      # losses['selfsup'].requires_grad = True
      if plot_dir is not None:
        uflow_plotting.plot_selfsup(key, images, flows, teacher_flow,
                                    student_flow, error, teacher_mask,
                                    student_mask, mask, selfsup_transform_fns,
                                    plot_dir)

  losses['total'] = sum(losses.values())

  return losses


def supervised_loss(weights, ground_truth_flow, ground_truth_valid,
                    predicted_flows):
  """Returns a supervised l1 loss when ground-truth flow is provided."""
  losses = {}
  # ground truth flow is given from image 0 to image 1
  predicted_flow = predicted_flows[(0, 1, 'augmented')][0]
  # resize flow to match ground truth (only changes resolution if ground truth
  # flow was not resized during loading (resize_gt_flow=False)
  _, _, height, width = list(ground_truth_flow.shape)
  # _, height, width, _ = ground_truth_flow.get_shape().as_list()
  predicted_flow = resize(predicted_flow, height, width, is_flow=True)
  # compute error/loss metric
  error = robust_l1(ground_truth_flow - predicted_flow)
  if ground_truth_valid is None:
    b, _, h, w = list(ground_truth_flow.shape)
    # b, h, w, _ = ground_truth_flow.shape.as_list()
    ground_truth_valid = torch.ones((b, 1, h, w), torch.float32)
  losses['supervision'] = (
      weights['supervision'] *
      (ground_truth_valid * error).sum() /
      ((ground_truth_valid).sum() + 1e-16))
  losses['total'] = losses['supervision']

  return losses


def compute_features_and_flow(
    feature_model,
    flow_model,
    batch,
    batch_without_aug,
    training,
    build_selfsup_transformations=None,
    teacher_feature_model=None,
    teacher_flow_model=None,
    teacher_image_version='original',
):
  """Compute features and flow for an image batch.

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    batch_without_aug: Batch without photometric augmentation
    training: bool that tells the model to use training or inference code.
    build_selfsup_transformations: A function which, when called with images
      and flows, populates the images and flows dictionary with student images
      and modified teacher flows corresponding to the student images.
    teacher_feature_model: None or instance of of feature model. If None, will
      not compute features and images for teacher distillation.
    teacher_flow_model: None or instance of flow model. If None, will not
      compute features and images for teacher distillation.
    teacher_image_version: str, either 'original' or 'augmented'

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """

  images = dict()
  flows = dict()
  features = dict()

  seq_len = int(batch.shape[1])

  perform_selfsup = (
      training and teacher_feature_model is not None and
      teacher_flow_model is not None and
      build_selfsup_transformations is not None)
  if perform_selfsup:
    selfsup_transform_fns = build_selfsup_transformations()
  else:
    selfsup_transform_fns = None

  for i in range(seq_len):
    # Populate teacher images with native, unmodified images.
    images[(i, 'original')] = batch_without_aug[:, i]
    images[(i, 'augmented')] = batch[:, i]
    if perform_selfsup:
      images[(i, 'transformed')] = selfsup_transform_fns[0](
          images[(i, 'augmented')], i_or_ij=i, is_flow=False)

  for key, image in images.items():
    i, image_version = key
    # if perform_selfsup and image_version == 'original':
    if perform_selfsup and image_version == teacher_image_version:
      teacher_feature_model.eval()
      features[(i, 'original-teacher')] = teacher_feature_model(
          image, split_features_by_sample=False)

    if training:
      features[(i, image_version + '-student')] = feature_model(
        image, split_features_by_sample=False)
    else:
        feature_model.eval()
        features[(i, image_version + '-student')] = feature_model(
          image, split_features_by_sample=False)


  # Only use original images and features computed on those for computing
  # photometric losses down the road.
  images = {i: images[(i, 'original')] for i in range(seq_len)}

  # Compute flow for all pairs of consecutive images that have the same (or no)
  # transformation applied to them, i.e. that have the same t.
  # pylint:disable=dict-iter-missing-items
  for (i, ti) in features:
    for (j, tj) in features:
      if (i + 1 == j or i - 1 == j) and ti == tj:
        t = ti
        key = (i, j, t)
        # No need to compute the flow for student applied to the original
        # image. We just need the features from that for the photometric loss.
        if t in ['augmented-student', 'transformed-student']:
          # Compute flow from i to j, defined in image i.
          flow = flow_model(
              features[(i, t)], features[(j, t)], training=training)

        elif t in ['original-teacher']:
          flow = teacher_flow_model(
              features[(i, t)], features[(j, t)], training=False)
        else:
          continue

        # Keep flows at levels 0-2.
        flow_level2 = flow[0]
        flow_level1 = upsample(flow_level2, is_flow=True)
        flow_level0 = upsample(flow_level1, is_flow=True)
        flows[key] = [flow_level0, flow_level1, flow_level2]

  return flows, selfsup_transform_fns


def compute_flow_for_supervised_loss(feature_model, flow_model, batch,
                                     training):
  """Compute features and flow for an image batch.

  Args:
    feature_model: A model to compute features for flow.
    flow_model: A model to compute flow.
    batch: A tf.tensor of shape [b, seq, h, w, c] holding a batch of triplets.
    training: bool that tells the model to use training or inference code.

  Returns:
    A tuple consisting of the images, the extracted features, the estimated
    flows, and the upsampled refined flows.
  """

  flows = dict()

  image_0 = batch[:, 0]
  image_1 = batch[:, 1]

  features_0 = feature_model(
      image_0, split_features_by_sample=False, training=training)
  features_1 = feature_model(
      image_1, split_features_by_sample=False, training=training)

  flow = flow_model(features_0, features_1, training=training)
  flow_level2 = flow[0]
  flow_level1 = upsample(flow_level2, is_flow=True)
  flow_level0 = upsample(flow_level1, is_flow=True)
  flows[(0, 1, 'augmented')] = [flow_level0, flow_level1, flow_level2]

  return flows


def random_crop(batch, max_offset_height=32, max_offset_width=32):
  """Randomly crop a batch of images.

  Args:
    batch: a 4-D tensor of shape [batch_size, height, width, num_channels].
    max_offset_height: an int, the maximum vertical coordinate of the top left
      corner of the cropped result.
    max_offset_width: an int, the maximum horizontal coordinate of the top left
      corner of the cropped result.

  Returns:
    a pair of 1) the cropped images in form of a tensor of shape
    [batch_size, height-max_offset, width-max_offset, num_channels],
    2) an offset tensor of shape [batch_size, 2] for height and width offsets.
  """

  # Compute current shapes and target shapes of the crop.
  batch_size, height, width, num_channels = batch.shape
  target_height = height - max_offset_height
  target_width = width - max_offset_width

  # Randomly sample offsets.
  offsets_height = (0 - (max_offset_height + 1)) * torch.rand([batch_size]) + (max_offset_height + 1)
  offsets_height = offsets_height.type(torch.int32)
  offsets_width = (0 - (max_offset_width + 1)) * torch.rand([batch_size]) + (max_offset_width + 1)
  offsets_width = offsets_width.type(torch.int32)

  offsets = torch.stack([offsets_height, offsets_width], dim=-1)

  # Loop over the batch and perform cropping.
  cropped_images = []
  for image, offset_height, offset_width in zip(batch, offsets_height,
                                                offsets_width):
    cropped_images.append(
      image[offset_height:offset_height + target_height, offset_width:offset_width + target_width, 0:num_channels])

  cropped_batch = torch.stack(cropped_images)

  return cropped_batch, offsets

def random_shift(batch, max_shift_height=32, max_shift_width=32):
  """Randomly shift a batch of images (with wrap around).

  Args:
    batch: a 4-D tensor of shape [batch_size, height, width, num_channels].
    max_shift_height: an int, the maximum shift along the height dimension in
      either direction.
    max_shift_width: an int, the maximum shift along the width dimension in
      either direction

  Returns:
    a pair of 1) the shifted images in form of a tensor of shape
    [batch_size, height, width, num_channels] and 2) the random shifts of shape
    [batch_size, 2], where positive numbers mean the image was shifted
    down / right and negative numbers mean it was shifted up / left.
  """

  # Randomly sample by how much the images are being shifted.
  batch_size, _, _, _ = batch.shape
  shifts_height = (-max_shift_height -(max_shift_height + 1))* torch.rand([batch_size]) + (max_shift_height + 1)
  shifts_height = shifts_height.type(torch.int32)

  shifts_width = (-max_shift_width - (max_shift_width + 1)) * torch.rand([batch_size]) + (max_shift_width + 1)
  shifts_width = shifts_width.type(torch.int32)


  shifts = torch.stack([shifts_height, shifts_width], dim=-1)

  # Loop over the batch and shift the images
  shifted_images = []
  for image, shift_height, shift_width in zip(batch, shifts_height,
                                              shifts_width):
    shifted_images.append(
        torch.roll(image, shifts=[shift_height, shift_width], dims=[0, 1]))
  shifted_images = torch.stack(shifted_images)

  return shifted_images, shifts


def randomly_shift_features(feature_pyramid,
                            max_shift_height=64,
                            max_shift_width=64):
  """Randomly shift a batch of images (with wrap around).

  Args:
    feature_pyramid: a list of 4-D tensors of shape [batch_size, height, width,
      num_channels], where the first entry is at level 1 (image size / 2).
    max_shift_height: an int, the maximum shift along the height dimension in
      either direction.
    max_shift_width: an int, the maximum shift along the width dimension in
      either direction

  Returns:
    a pair of 1) a list of shifted feature images as tensors of shape
    [batch_size, height, width, num_channels] and 2) the random shifts of shape
    [batch_size, 2], where positive numbers mean the image was shifted
    down / right and negative numbers mean it was shifted up / left.
  """
  batch_size, height, width = feature_pyramid[0].shape[:3]
  # Image size is double the size of the features at level1 (index 0).
  height *= 2
  width *= 2

  # Transform the shift range to the size of the top level of the pyramid.
  top_level_scale = 2**len(feature_pyramid)
  max_shift_height_top_level = max_shift_height // top_level_scale
  max_shift_width_top_level = max_shift_width // top_level_scale

  # Randomly sample by how much the images are being shifted at the top level
  # and scale the shift back to level 0 (original image resolution).
  shifts_height = top_level_scale *((-max_shift_height_top_level - (max_shift_height_top_level + 1))*torch.rand([batch_size]) + (max_shift_height_top_level + 1))
  shifts_height = shifts_height.type(torch.int32)

  shifts_width = top_level_scale *((-max_shift_width_top_level - (max_shift_width_top_level + 1))*torch.rand([batch_size]) + (max_shift_width_top_level + 1))
  shifts_width = shifts_width.type(torch.int32)

  shifts = torch.stack([shifts_height, shifts_width], dim=-1)

  # Iterate over pyramid levels.
  shifted_features = []
  for level, feature_image_batch in enumerate(feature_pyramid, start=1):
    shifts_at_this_level = shifts // 2**level
    # pylint:disable=g-complex-comprehension
    shifted_features.append(
        torch.stack([
            torch.roll(
                feature_image_batch[i],
                shifts=shifts_at_this_level[i],
                dims=[0, 1]) for i in range(batch_size)
        ],
                 dim=0))

  return shifted_features, shifts.type(torch.float32)


def zero_mask_border(mask_bhw3, patch_size):
  """Used to ignore border effects from census_transform."""
  mask_padding = patch_size // 2
  mask = mask_bhw3[:,:, mask_padding:-mask_padding, mask_padding:-mask_padding]

  return torch.nn.functional.pad(mask, ( mask_padding, mask_padding,
                mask_padding, mask_padding), mode='constant', value=0)



def census_transform(image, patch_size):
  """The census transform as described by DDFlow.

  See the paper at https://arxiv.org/abs/1902.09145

  Args:
    image: tensor of shape (b, h, w, c)
    patch_size: int
  Returns:
    image with census transform applied
  """

  rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
  intensities = image[:, 0, Ellipsis] * rgb_weights[0] + image[:, 1, Ellipsis] * rgb_weights[1] + image[:, 2, Ellipsis] * rgb_weights[2]
  intensities = torch.unsqueeze(intensities, 1) * 255
  kernel = torch.reshape(
    torch.eye(patch_size * patch_size),
    (patch_size * patch_size, 1, patch_size, patch_size)).to(uflow_gpu_utils.device)

  # intensities = torch.moveaxis(intensities,-1,1)
  # kernel = torch.moveaxis(kernel,-1,1)
  neighbors = F.conv2d(input= intensities, weight= kernel,
                       padding=(int((patch_size - 1) / 2), int((patch_size - 1) / 2)))
  # neighbors = torch.moveaxis(neighbors, 1, -1)

  diff = neighbors - intensities
  # Coefficients adopted from DDFlow.
  diff_norm = diff / torch.sqrt(.81 + torch.square(diff))
  return diff_norm


def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
  """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.

  Args:
    a_bhwk: tf.Tensor of shape (batch, height, width, features)
    b_bhwk: tf.Tensor of shape (batch, height, width, features)
    thresh: float threshold

  Returns:
    a tensor with approx. 1 in (h, w) locations that are significantly
    more different than thresh and approx. 0 if significantly less
    different than thresh.
  """
  sq_dist_bhwk = torch.square(a_bhwk - b_bhwk)
  soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
  return soft_thresh_dist_bhwk.sum(dim = 1, keepdim = True)



def census_loss(image_a_bhw3,
                image_b_bhw3,
                mask_bhw3,
                patch_size=7,
                distance_metric_fn=abs_robust_loss):
  """Compares the similarity of the census transform of two images."""
  census_image_a_bhwk = census_transform(image_a_bhw3, patch_size)
  census_image_b_bhwk = census_transform(image_b_bhw3, patch_size)

  hamming_bhw1 = soft_hamming(census_image_a_bhwk, census_image_b_bhwk)

  # Set borders of mask to zero to ignore edge effects.
  padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)
  diff = distance_metric_fn(hamming_bhw1)
  diff *= padded_mask_bhw3
  diff_sum = diff.sum()
  # padded_mask_bhw3.requires_grad = False
  padded_mask_bhw3.detach()
  loss_mean = diff_sum / (( padded_mask_bhw3 + 1e-6).sum())

  return loss_mean


def _avg_pool3x3(x):
  return torch.nn.functional.avg_pool2d(x, (3,3), (1,1))



def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
  """Computes a weighted structured image similarity measure.

  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.

  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.

  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
  if c1 == float('inf') and c2 == float('inf'):
    raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                     'likely unintended.')
  weight = torch.unsqueeze(weight, 1)
  average_pooled_weight = _avg_pool3x3(weight)
  weight_plus_epsilon = weight + weight_epsilon
  inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

  def weighted_avg_pool3x3(z):
    wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
    return wighted_avg * inverse_average_pooled_weight

  mu_x = weighted_avg_pool3x3(x)
  mu_y = weighted_avg_pool3x3(y)
  sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
  sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
  sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
  if c1 == float('inf'):
    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
  elif c2 == float('inf'):
    ssim_n = 2 * mu_x * mu_y + c1
    ssim_d = mu_x**2 + mu_y**2 + c1
  else:
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  result = ssim_n / ssim_d
  return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


def compute_f_metrics(mask_prediction, mask_gt, num_thresholds=40):
  """Return a dictionary of the true positives, etc. for two binary masks."""
  results = defaultdict(dict)
  mask_prediction = mask_prediction.type(torch.float32)
  mask_gt = mask_gt.type(torch.float32)
  for threshold in np.linspace(0, 1, num_thresholds):
    mask_thresh = torch.greater(mask_prediction, threshold).type(torch.float32)
    true_pos = torch.count_nonzero(mask_thresh * mask_gt).type(torch.float32)
    true_neg = torch.count_nonzero((mask_thresh - 1) * (mask_gt - 1))
    false_pos = torch.count_nonzero(mask_thresh * (mask_gt - 1)).type(torch.float32)
    false_neg = torch.count_nonzero((mask_thresh - 1) * mask_gt).type(torch.float32)
    results[threshold]['tp'] = true_pos
    results[threshold]['fp'] = false_pos
    results[threshold]['fn'] = false_neg
    results[threshold]['tn'] = true_neg
  return results

def time_it(f, num_reps=1, execute_once_before=False):
  """Times a tensorflow function in eager mode.

  Args:
    f: function with no arguments that should be timed.
    num_reps: int, number of repetitions for timing.
    execute_once_before: boolean, whether to execute the function once before
      timing in order to not count the tf.function compile time.

  Returns:
    tuple of the average time in ms and the functions output.
  """
  assert num_reps >= 1
  # Execute f once before timing it to allow tf.function to compile the graph.
  if execute_once_before:
    x = f()
  # Make sure that there is nothing still running on the GPU by waiting for the
  # completion of a bogus command.
  _ = torch.square(torch.rand([1])).numpy()
  # Time f for a number of repetitions.
  start_in_s = time.time()
  for _ in range(num_reps):
    x = f()
    # Make sure that f has finished and was not just enqueued by using another
    # bogus command. This will overestimate the computing time of f by waiting
    # until the result has been copied to main memory. Calling reduce_sum
    # reduces that overestimation.
    if isinstance(x, tuple) or isinstance(x, list):
      _ = [torch.sum(xi) for xi in x]
    else:
      _ = torch.sum(x)
  end_in_s = time.time()
  # Compute the average time in ms.
  avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
  return avg_time, x

def get_fmax_and_best_thresh(results):
  """Select which threshold produces the best f1 score."""
  fmax = -1.
  best_thresh = -1.
  for thresh, metrics in results.items():
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    if f1 > fmax:
      fmax = f1
      best_thresh = thresh
  return fmax, best_thresh