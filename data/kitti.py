import os
from absl import  flags
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random

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