import os
from absl import  flags
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import time
import  numpy as np
from collections import defaultdict
import sys
from torch.utils.tensorboard import SummaryWriter

import uflow_flags
import uflow_utils
from uflow_utils import resize
import uflow_gpu_utils

FLAGS = flags.FLAGS

class CustomDataSet(Dataset):
    def __init__(self, main_dir,  transform, entire_seq = False, height= None, width= None ,resize_gt_flow=True):
        self.main_dir = main_dir
        self.transform = transform
        self.height = height
        self.width = width
        self.entire_sequence = entire_seq
        self.total_imgs = sorted(os.listdir(main_dir))
        self.num_sequence = int(self.total_imgs[-1][:-7])
        self.data_tuples = []
        # i = 0
        # while i != len(self.total_imgs) -1 :
        #     self.data_tuples.append([self.total_imgs[i], self.total_imgs[i + 1]])            #written for a sequence length of two
        #     i += 1
        #     if i != 0 and i % 20 == 0:
        #         i = i + 1
        #
        if self.entire_sequence:
            sequences = [list(range(21))]
        else:
            # Of the 21 frames, ignore frames 9-12 because we will test on those.
            sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                         [13, 14, 15, 16, 17, 18, 19, 20]]
        for i in range(self.num_sequence):
            for js in sequences:
                image_files = ['%06d_%02d.png' % (i, j) for j in js]
                image_tuples = [[image_files[i], image_files[i + 1]] for i in range(len(image_files) - 1)]
                self.data_tuples.append(image_tuples)

        # random.shuffle(self.data_tuples)
        self.data_tuples = [pair for seq in self.data_tuples for pair in seq]

        self.height = height
        self.width = width
        self.resize_gt_flow = resize_gt_flow


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir, self.data_tuples[idx][0])
        image1 = Image.open(img_loc1).convert("RGB")
        tensor_image1 = self.transform(image1).to(uflow_gpu_utils.device)
        tensor_image1 = resize(tensor_image1, height= self.height, width= self.width, is_flow= False)

        img_loc2 = os.path.join(self.main_dir, self.data_tuples[idx][1])
        image2 = Image.open(img_loc2).convert("RGB")
        tensor_image2 = self.transform(image2).to(uflow_gpu_utils.device)
        tensor_image2 = resize(tensor_image2, height=self.height, width=self.width, is_flow=False)
        # tensor_image = uflow_utils.resize(tensor_image, height= self.height, width= self.width, is_flow= False)

        return torch.stack([tensor_image1, tensor_image2], dim= 0)


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 entire_seq = False,
                 seed=41):
# path = "/home/manish/winshare/datasets/data_scene_flow_multiview"

    if ',' in path:
        l = path.split(',')
        d = '/'.join(l[0].split('/')[:-1])
        l[0] = l[0].split('/')[-1]
        paths = [os.path.join(d, x) for x in l]
    else:
        paths = path

    image_dir = os.path.join(paths, "training" + '/image_2')   #os.path.realpath('..'),

    dataset = CustomDataSet(image_dir, transform= transforms.ToTensor(), entire_seq= entire_seq, height= height, width= width)

    return dataset


class CustomDataSetEval(Dataset):
    def __init__(self, main_dir, flow_noc_dir, flow_occ_dir,  transform,  height= None, width= None ,resize_gt_flow=True):
        self.main_dir = main_dir
        self.noc_dir = flow_noc_dir
        self.occ_dir = flow_occ_dir
        self.transform = transform
        self.height = height
        self.width = width

        self.total_imgs = sorted(os.listdir(main_dir))
        self.num_sequence = int(self.total_imgs[-1][:-7])
        self.data_tuples = []

        for i in range(self.num_sequence):
            image_tuples = ['%06d_%02d.png' % (i, j) for j in [10,11]]
            self.data_tuples.append(image_tuples)

        self.height = height
        self.width = width
        self.resize_gt_flow = resize_gt_flow


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir, self.data_tuples[idx][0])
        image1 = Image.open(img_loc1).convert("RGB")
        tensor_image1 = self.transform(image1).to(uflow_gpu_utils.device)
        # tensor_image1 = resize(tensor_image1, height= self.height, width= self.width, is_flow= False)

        img_loc2 = os.path.join(self.main_dir, self.data_tuples[idx][1])
        image2 = Image.open(img_loc2).convert("RGB")
        tensor_image2 = self.transform(image2).to(uflow_gpu_utils.device)
        # tensor_image2 = resize(tensor_image2, height=self.height, width=self.width, is_flow=False)

        flow_occ_loc = os.path.join(self.occ_dir, self.data_tuples[idx][0])
        flow_occ = Image.open(flow_occ_loc)
        flow_occ = np.array(flow_occ, dtype=np.uint16)
        flow_uv_occ = (flow_occ[Ellipsis, :2].astype(np.float32) - 2 ** 15) / 64.0
        flow_uv_occ = np.rollaxis(flow_uv_occ, -1, 0)
        flow_valid_occ = flow_occ[Ellipsis, 2:3].astype(np.uint8)
        flow_valid_occ = np.rollaxis(flow_valid_occ, -1, 0)
        flow_uv_occ = torch.from_numpy(flow_uv_occ)
        flow_valid_occ = torch.from_numpy(flow_valid_occ)

        flow_noc_loc = os.path.join(self.noc_dir, self.data_tuples[idx][0])
        flow_noc = Image.open(flow_noc_loc)
        flow_noc = np.array(flow_noc, dtype=np.uint16)
        flow_uv_noc = (flow_noc[Ellipsis, :2].astype(np.float32) - 2 ** 15) / 64.0
        flow_uv_noc = np.rollaxis(flow_uv_noc, -1, 0)
        flow_valid_noc = flow_noc[Ellipsis, 2:3].astype(np.uint8)
        flow_valid_noc = np.rollaxis(flow_valid_noc, -1, 0)
        flow_uv_noc = torch.from_numpy(flow_uv_noc)
        flow_valid_noc = torch.from_numpy(flow_valid_noc)

        return torch.stack([tensor_image1, tensor_image2], dim= 0) , flow_uv_occ, flow_uv_noc, flow_valid_occ, flow_valid_noc

def make_eval_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 entire_seq = False,
                 seed=41):

    if height is not None or width is not None:
      raise ValueError('for_eval is incompatible with height/width')
    if shuffle_buffer_size:
      raise ValueError('for_eval is incompatible with shuffle_buffer_size')
    if seq_len != 2:
      raise ValueError('for_eval only compatible with seq_len == 2.')

    if ',' in path:
        l = path.split(',')
        d = '/'.join(l[0].split('/')[:-1])
        l[0] = l[0].split('/')[-1]
        paths = [os.path.join(d, x) for x in l]
    else:
        paths = path

    image_dir = os.path.join(paths, "training" + '/image_2')
    flow_noc_dir = os.path.join(paths, "training" + "/flow_noc")
    flow_occ_dir = os.path.join(paths, "training" + "/flow_occ")

    dataset = CustomDataSetEval(image_dir, flow_noc_dir= flow_noc_dir, flow_occ_dir = flow_occ_dir, transform=transforms.ToTensor(),  height=height,  width=width)

    return dataset

def evaluate(inference_fn,
             dataset,
             height,
             width,
             progress_bar=False,
             plot_dir='',
             num_plots=0,
             prefix='kitti'):
  """Evaluate an iference function for flow with a kitti eval dataset.

  Args:
    inference_fn: An inference function that produces a flow_field from two
      images, e.g. the infer method of UFlow.
    dataset: A dataset produced by the method above with for_eval=True.
    height: int, the height to which the images should be resized for inference.
    width: int, the width to which the images should be resized for inference.
    progress_bar: boolean, flag to indicate whether the function should print a
      progress_bar during evaluaton.
    plot_dir: string, optional path to a directory in which plots are saved (if
      num_plots > 0).
    num_plots: int, maximum number of qualitative results to plot for the
      evaluation.
    prefix: str to prefix evaluation keys with in the returned dictionary.

  Returns:
    A dictionary of floats that represent different evaluation metrics. The keys
    of this dictionary are returned by the method list_eval_keys (see below).
  """

  eval_start_in_s = time.time()

  it = data.DataLoader(dataset)
  # it = tf.compat.v1.data.make_one_shot_iterator(dataset)
  epe_occ = []  # End point errors.
  errors_occ = []
  valid_occ = []
  epe_noc = []  # End point errors.
  errors_noc = []
  valid_noc = []
  inference_times = []
  all_occlusion_results = defaultdict(lambda: defaultdict(int))



  for i, test_batch in enumerate(it):

    if progress_bar:
      sys.stdout.write(':')
      sys.stdout.flush()
    (image_batch, flow_uv_occ, flow_uv_noc, flow_valid_occ,
     flow_valid_noc) = test_batch

    flow_valid_occ = flow_valid_occ.type(torch.float32)
    flow_valid_noc = flow_valid_noc.type(torch.float32)


  # pylint:disable=cell-var-from-loop
    f = lambda: inference_fn(
        image_batch[0],       #change back to 0 here and below to 1. Trying to get the backward flow
        image_batch[1],
        input_height=height,
        input_width=width,
        infer_occlusion=True)
    inference_time_in_ms, (flow, soft_occlusion_mask) = uflow_utils.time_it(
        f, execute_once_before=i == 0)
    inference_times.append(inference_time_in_ms)

    occ_mask_gt = flow_valid_occ - flow_valid_noc
    f_dict = uflow_utils.compute_f_metrics(soft_occlusion_mask * flow_valid_occ,
                                          occ_mask_gt * flow_valid_occ)
    best_thresh = -1.
    best_f_score = -1.
    for thresh, metrics in f_dict.items():
      precision = metrics['tp'] / (metrics['tp'] + metrics['fp'] + 1e-6)
      recall = metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-6)
      f1 = 2 * precision * recall / (precision + recall + 1e-6)
      if f1 > best_f_score:
        best_thresh = thresh
        best_f_score = f1
      all_occlusion_results[thresh]['tp'] += metrics['tp']
      all_occlusion_results[thresh]['fp'] += metrics['fp']
      all_occlusion_results[thresh]['tn'] += metrics['tn']
      all_occlusion_results[thresh]['fn'] += metrics['fn']

    mask_thresh = torch.greater(soft_occlusion_mask, best_thresh).type(torch.float32)
    # Image coordinates are swapped in labels
    final_flow = flow.numpy()
    final_flow = final_flow[::-1, Ellipsis]
    final_flow = torch.from_numpy(final_flow)


    endpoint_error_occ = torch.sum((final_flow - flow_uv_occ)**2, dim= 1, keepdim=True)**0.5
    gt_flow_abs = torch.sum(flow_uv_occ**2, dim= 1, keepdim=True)**0.5
    outliers_occ = torch.logical_and(endpoint_error_occ > 3.,
                       endpoint_error_occ > 0.05 * gt_flow_abs).type(torch.float32)

    endpoint_error_noc = torch.sum((final_flow - flow_uv_noc)**2, dim=1, keepdim=True)**0.5
    gt_flow_abs = torch.sum(flow_uv_noc**2, dim= 1, keepdim=True)**0.5
    outliers_noc = torch.logical_and(endpoint_error_noc > 3.,
                       endpoint_error_noc > 0.05 * gt_flow_abs).type(torch.float32)

    epe_occ.append(torch.sum(flow_valid_occ * endpoint_error_occ))
    errors_occ.append(torch.sum(flow_valid_occ * outliers_occ))
    valid_occ.append(torch.sum(flow_valid_occ))

    epe_noc.append(
        torch.sum(flow_valid_noc * endpoint_error_noc))
    errors_noc.append(torch.sum(flow_valid_noc * outliers_noc))
    valid_noc.append(torch.sum(flow_valid_noc))

    if plot_dir and i < num_plots:
      uflow_plotting.complete_paper_plot(
          plot_dir,
          i,
          image_batch[0].numpy(),
          image_batch[1].numpy(),
          final_flow.numpy(),
          flow_uv_occ.numpy(),
          flow_valid_occ.numpy(), (1. - mask_thresh).numpy(),
          (1. - occ_mask_gt).numpy(),
          frame_skip=None)
  if progress_bar:
    sys.stdout.write('\n')
    sys.stdout.flush()

  fmax, best_thresh = uflow_utils.get_fmax_and_best_thresh(all_occlusion_results)
  eval_stop_in_s = time.time()

  results = {
      prefix + '-occl-f-max':
          fmax,
      prefix + '-best-occl-thresh':
          best_thresh,
      prefix + '-EPE(occ)':
          np.clip(np.mean(np.array(epe_occ) / np.array(valid_occ)), 0.0, 50.0),
      prefix + '-ER(occ)':
          np.mean(np.array(errors_occ) / np.array(valid_occ)),
      prefix + '-EPE(noc)':
          np.clip(np.mean(np.array(epe_noc) / np.array(valid_noc)), 0.0, 50.0),
      prefix + '-ER(noc)':
          np.mean(np.array(errors_noc) / np.array(valid_noc)),
      prefix + '-inf-time(ms)':
          np.mean(inference_times),
      prefix + '-eval-time(s)':
          eval_stop_in_s - eval_start_in_s,
  }

  writer = SummaryWriter(log_dir=FLAGS.summary_dir)
  for key, val in results.items():
      step = uflow_flags.step
      writer.add_scalar(key, val, step)

  return results


def list_eval_keys(prefix='kitti'):
  """List the keys of the dictionary returned by the evaluate function."""
  return [
      prefix + '-EPE(occ)',
      prefix + '-EPE(noc)',
      prefix + '-ER(occ)',
      prefix + '-ER(noc)',
      prefix + '-inf-time(ms)',
      prefix + '-eval-time(s)',
      prefix + '-occl-f-max',
      prefix + '-best-occl-thresh',
  ]



