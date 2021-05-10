from collections import defaultdict
import os
import sys
import time


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from absl import  flags
from PIL import Image

FLAGS = flags.FLAGS

class CustomDataSet(Dataset):
    def __init__(self, main_dir,  transform, height= None, width= None ,resize_gt_flow=True):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = sorted(os.listdir(main_dir))
        self.data_tuples = []
        i = 0
        while i != len(self.total_imgs) -1 :
            if i != 0 and i % 20 == 0:
                i = i + 1
            self.data_tuples.append([self.total_imgs[i], self.total_imgs[i + 1]])            #written for a sequence length of two
            i += 1

        self.height = height
        self.width = width
        self.resize_gt_flow = resize_gt_flow


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir, self.data_tuples[idx][0])
        image1 = Image.open(img_loc1).convert("RGB")
        tensor_image1 = self.transform(image1)

        img_loc2 = os.path.join(self.main_dir, self.data_tuples[idx][1])
        image2 = Image.open(img_loc2).convert("RGB")
        tensor_image2 = self.transform(image2)
        # tensor_image = uflow_utils.resize(tensor_image, height= self.height, width= self.width, is_flow= False)

        return torch.stack([tensor_image1, tensor_image2], dim= 0)


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 seed=41):
# path = "/home/manish/winshare/datasets/data_scene_flow_multiview"

    if ',' in path:
        l = path.split(',')
        d = '/'.join(l[0].split('/')[:-1])
        l[0] = l[0].split('/')[-1]
        paths = [os.path.join(d, x) for x in l]
    else:
        paths = path

    image_dir = os.path.join(os.path.realpath('..'), paths, "training" + '/image_2')

    dataset = CustomDataSet(image_dir, transform= transforms.ToTensor(), height= height, width= width)

    return dataset


























  # Generate list of filenames.
  # pylint:disable=g-complex-comprehension
  # files = [os.path.join(d, f) for d in paths for f in tf.io.gfile.listdir(d)]
  # files = [os.path.join(d, f) for d in paths for f in sorted(os.listdir(d))]
  # num_files = len(files)
  #
  # if 'train' in mode:
  #   rgen = np.random.RandomState(seed)
  #   rgen.shuffle(files)
  # ds = tf.data.Dataset.from_tensor_slices(files)
#
#   if mode == 'eval':
#     if height is not None or width is not None:
#       raise ValueError('for_eval is incompatible with height/width')
#     if shuffle_buffer_size:
#       raise ValueError('for_eval is incompatible with shuffle_buffer_size')
#     if seq_len != 2:
#       raise ValueError('for_eval only compatible with seq_len == 2.')
#     ds = ds.map(tf.data.TFRecordDataset)
#     # Parse each element of the subsequences and unbatch the result.
#     # pylint:disable=g-long-lambda
#     ds = ds.interleave(
#         lambda x: x.map(
#             parse_eval_data,
#             num_parallel_calls=tf.data.experimental.AUTOTUNE),
#         cycle_length=min(10, num_files),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#   elif mode == 'test':
#     if height is not None or width is not None:
#       raise ValueError('for_eval is incompatible with height/width')
#     if shuffle_buffer_size:
#       raise ValueError('for_eval is incompatible with shuffle_buffer_size')
#     if seq_len != 2:
#       raise ValueError('for_eval only compatible with seq_len == 2.')
#     ds = ds.map(tf.data.TFRecordDataset)
#     # Parse each element of the subsequences and unbatch the result.
#     # pylint:disable=g-long-lambda
#     ds = ds.flat_map(lambda x: x.map(
#         parse_test_data,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE))
#   elif mode == 'train' or mode == 'video':
#     if shuffle_buffer_size:
#       ds = ds.shuffle(num_files)
#     # Create a nested dataset.
#     ds = ds.map(tf.data.TFRecordDataset)
#     # Parse each element of the subsequences and unbatch the result.
#     # pylint:disable=g-long-lambda
#     ds = ds.map(lambda x: x.map(
#         lambda y: parse_data(y, height, width),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch())
#     # Slide a window over each dataset, combine either by interleaving or by
#     # sequencing the result (produces a a nested dataset)
#     window_fn = lambda x: x.window(size=seq_len, shift=1, drop_remainder=True)
#     # Interleave subsequences (too long cycle length causes memory issues).
#     ds = ds.interleave(
#         window_fn,
#         cycle_length=1 if 'video' in mode else min(10, num_files),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     if shuffle_buffer_size:
#       # Shuffle subsequences.
#       ds = ds.shuffle(buffer_size=shuffle_buffer_size)
#
#     # Put repeat after shuffle.
#     ds = ds.repeat()
#     # Flatten the nested dataset into a batched dataset.
#     ds = ds.flat_map(lambda x: x.batch(seq_len))
#     # Prefetch a number of batches because reading new ones can take much longer
#     # when they are from new files.
#     ds = ds.prefetch(10)
#   elif 'train' in mode and 'sup' in mode:
#     if shuffle_buffer_size:
#       ds = ds.shuffle(num_files)
#     # Create a nested dataset.
#     ds = ds.map(tf.data.TFRecordDataset)
#     # Parse each element of the subsequences and unbatch the result.
#     ds = ds.interleave(
#         lambda x: x.map(
#             lambda y: parse_supervised_train_data(y, height, width,
#                                                   resize_gt_flow),
#             num_parallel_calls=tf.data.experimental.AUTOTUNE),
#         cycle_length=min(10, num_files),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     if shuffle_buffer_size:
#       # Shuffle subsequences.
#       ds = ds.shuffle(buffer_size=shuffle_buffer_size)
#
#     # Put repeat after shuffle.
#     ds = ds.repeat()
#     # Prefetch a number of batches because reading new ones can take much longer
#     # when they are from new files.
#     ds = ds.prefetch(10)
#
#
#
#   return ds