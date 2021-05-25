import io
import os
import time

import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

# import flowpy


_FLOW_SCALING_FACTOR = 50.0

def print_log(log, epoch=None, mean_over_num_steps=1):
  """Print log returned by UFlow.train(...)."""

  if epoch is None:
    status = ''
  else:
    status = '{} -- '.format(epoch)

  status += 'total-loss: {:.6f}'.format(
      np.mean(log['total-loss'][-mean_over_num_steps:]))

  for key in sorted(log):
    if key not in ['total-loss']:
      loss_mean = np.mean(log[key][-mean_over_num_steps:])
      status += ', {}: {:.6f}'.format(key, loss_mean)
  print(status)

def print_eval(eval_dict):
  """Prints eval_dict to console."""

  status = ''.join(
      ['{}: {:.6f}, '.format(key, eval_dict[key]) for key in sorted(eval_dict)])
  print(status[:-2])


def plot_log(log, plot_dir):
  plt.figure(1)
  plt.clf()

  keys = ['total-loss'
         ] + [key for key in sorted(log) if key not in ['total-loss']]
  for key in keys:
    plt.plot(log[key], '--' if key == 'total-loss' else '-', label=key)
  plt.legend()
  save_and_close(os.path.join(plot_dir, 'log.png'))

def save_and_close(filename):
  """Save figures."""

  # Create a python byte stream into which to write the plot image.
  buf = io.BytesIO()

  # Save the image into the buffer.
  plt.savefig(buf, format='png')

  # Seek the buffer back to the beginning, then either write to file or stdout.
  buf.seek(0)
  with tf.io.gfile.GFile(filename, 'w') as f:
    f.write(buf.read(-1))
  plt.close('all')


def flow_to_rgb(flow):
  """Computes an RGB visualization of a flow field."""
  shape = list(flow.shape)
  # print(shape)

  height, width = [float(s) for s in shape[-2:]]
  scaling = _FLOW_SCALING_FACTOR / (height**2 + width**2)**0.5

  motion_angle = np.arctan2(flow[1, Ellipsis], flow[0, Ellipsis])
  motion_magnitude = (flow[1, Ellipsis]**2 + flow[0, Ellipsis]**2)**0.5
  # print(motion_angle.shape)
  # print(motion_magnitude.shape)
  # Visualize flow using the HSV color space, where angles are represented by
  # hue and magnitudes are represented by saturation.

  flow_hsv = np.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                         np.clip(motion_magnitude * scaling, 0.0, 1.0),
                         np.ones_like(motion_magnitude)],
                        axis= -1)

  # Transform colors from HSV to RGB color space for plotting.
  # print(flow_hsv.shape)
  return matplotlib.colors.hsv_to_rgb(flow_hsv)



def complete_paper_plot(plot_dir,
                        index,
                        image1,
                        image2,
                        flow_uv,
                        ground_truth_flow_uv,
                        flow_valid_occ,
                        predicted_occlusion,
                        ground_truth_occlusion,
                        frame_skip=None):
  """Plots rgb image, flow, occlusions, ground truth, all as separate images."""

  def save_fig(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = str(index) + '_' + name
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

#############here#######################
  # def robust_l1(x):
  #     """Robust L1 metric."""
  #     return (x ** 2 + 0.001 ** 2) ** 0.5
  #
  #
  # error = robust_l1(ground_truth_flow_uv - flow_uv)
  #
  # mask_non_zero =  ground_truth_flow_uv != 0
  # mask_zero = ground_truth_flow_uv == 0
  #
  # loss_gt = (tf.reduce_sum(error[mask_non_zero]) / (tf.reduce_sum(tf.cast(mask_non_zero, tf.float32)) + 1e-16))
  # loss_zero = (tf.reduce_sum(error[mask_zero]) / (tf.reduce_sum(tf.cast(mask_zero, tf.float32)) + 1e-16))
  #
  # # flowpy.flow_write(plot_dir + '/flow_gt'+ str(index)+".flo",ground_truth_flow_uv)
  # flowpy.flow_write(plot_dir + '/flow_pred_bkwd' + str(index) + ".flo", flow_uv)
  #
  # # print(flow_uv.shape)
  # fig, axis = plt.subplots(3,2)
  # fig.set_figheight(14)
  # fig.set_figwidth(14)
  # axis[0,0].imshow(image1)
  # axis[0,0].set_title("Image1")
  # axis[0, 1].imshow(image2)
  # axis[0, 1].set_title("Image2")
  # max_radius_f = flowpy.get_flow_max_radius(ground_truth_flow_uv)
  # axis[1, 0].imshow(flowpy.flow_to_rgb(ground_truth_flow_uv, flow_max_radius= max_radius_f))
  # axis[1, 0].set_title("Ground-truth Flow")
  # flowpy.attach_calibration_pattern(axis[1,1], flow_max_radius=max_radius_f)
  # max_radius_p = flowpy.get_flow_max_radius(flow_uv)
  # axis[2, 0].imshow(flowpy.flow_to_rgb(flow_uv, flow_max_radius=max_radius_p))
  # axis[2, 0].set_title("Predicted Flow")
  # axis[2,0].set_xlabel('l1 loss for gt pixels: {} \n l1 loss for zero pixels: {}'.format(loss_gt,loss_zero))
  # flowpy.attach_calibration_pattern(axis[2,1], flow_max_radius=max_radius_p)
  # # print(np.mean(ground_truth_flow_uv), np.mean(flow_uv))
  #
  # axis[2,1].imshow((1-predicted_occlusion[:, :, 0]) * 255, cmap='Greys')
  # axis[2,1].set_title("Predicted Occlusion")
  #
  # # plt.imshow(flowpy.flow_to_rgb(flow_uv))
  # # plt.savefig( plot_dir+'/pred_flow'+str(index))
  # # plt.imshow(flowpy.flow_to_rgb(ground_truth_flow_uv))
  # # plt.savefig( plot_dir+'/gt_flow'+ str(index))
  # # print(ground_truth_flow_uv.shape)
  # plt.imshow(image1)
  # plt.savefig(plot_dir + '/plots'+ str(index ))


#############till_here##########################
  flow_uv = -flow_uv[::-1,:,:]
  ground_truth_flow_uv = -ground_truth_flow_uv[::-1,:, :]
  plt.figure()
  plt.clf()

  plt.imshow(np.moveaxis(((image1 + image2) / 2.),0,-1))

  save_fig('image_rgb', plot_dir)
  # np.save("flow_pred"+plot_dir,flow_uv)
  plt.imshow(flow_to_rgb(flow_uv))
  save_fig('predicted_flow', plot_dir)
  # np.save("flow_gt" + plot_dir, ground_truth_flow_uv * flow_valid_occ)
  plt.imshow(flow_to_rgb(ground_truth_flow_uv * flow_valid_occ))
  save_fig('ground_truth_flow', plot_dir)

  endpoint_error = np.sum(
      (ground_truth_flow_uv - flow_uv)**2, axis= 0 , keepdims=True)**0.5

  plt.imshow(
      (endpoint_error * flow_valid_occ)[0],
      cmap='viridis',
      vmin=0,
      vmax=40)
  save_fig('flow_error', plot_dir)

  plt.imshow((predicted_occlusion[0]) * 255, cmap='Greys')
  save_fig('predicted_occlusion', plot_dir)

  plt.imshow((ground_truth_occlusion[0]) * 255, cmap='Greys')
  save_fig('ground_truth_occlusion', plot_dir)

  plt.close('all')