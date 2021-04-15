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

"""Tests for uflow_utils."""

from absl.testing import absltest
import numpy as np
import torch
import torchvision.transforms as transforms
import uflow_utils


class UflowUtilsTest(absltest.TestCase):

  def test_fb_consistency_no_occlusion(self):
    batch_size = 4
    height = 64
    width = 64
    # flows points right and up by 4
    flow_01 = np.ones((batch_size, height, width, 2)) * 4.
    # flow points left and down by 4
    perfect_flow_10 = -flow_01
    flow_01 = torch.tensor(flow_01.astype(np.float32))

    resize_transform = transforms.Compose([transforms.Resize((int(height / 2), int(width / 2)))])
    flow_01 = torch.moveaxis(flow_01, -1, 1)
    flow_01_level1 = resize_transform(flow_01) / 2.
    flow_01_level1 = torch.moveaxis(flow_01_level1, 1, -1)

    perfect_flow_10 = torch.tensor(perfect_flow_10.astype(np.float32))
    perfect_flow_10_level1 = -flow_01_level1
    flows = {}
    flows[(0, 1, 0)] = [flow_01, flow_01_level1]
    flows[(1, 0, 0)] = [perfect_flow_10, perfect_flow_10_level1]
    _, _, _, not_occluded_masks, _, _ = \
        uflow_utils.compute_warps_and_occlusion(
            flows, occlusion_estimation='brox')
    # assert that nothing is occluded
    is_ones_01 = np.equal(
        np.ones((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(0, 1, 0)][0][:, 4:-4, 4:-4, :]).all()
    is_ones_10 = np.equal(
        np.ones((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(1, 0, 0)][0][:, 4:-4, 4:-4, :]).all()
    self.assertTrue(is_ones_01)
    self.assertTrue(is_ones_10)

  def test_fb_consistency_with_occlusion(self):
    batch_size = 4
    height = 64
    width = 64
    # flows points right and up by 4
    flow_01 = np.ones((batch_size, height, width, 2)) * 4.
    # flow points left and down by 2
    imperfect_flow_10 = -flow_01 * .5
    flow_01 = torch.tensor(flow_01.astype(np.float32))
    resize_transform = transforms.Compose([transforms.Resize((int(height / 2), int(width / 2)))])
    flow_01 = torch.moveaxis(flow_01, -1, 1)
    flow_01_level1 = resize_transform(flow_01) / 2.
    flow_01_level1 = torch.moveaxis(flow_01_level1, 1, -1)

    imperfect_flow_10 = torch.tensor(imperfect_flow_10.astype(np.float32))
    imperfect_flow_10_level1 = -flow_01_level1 * .5
    flows = {}
    flows[(0, 1, 0)] = [flow_01, flow_01_level1]
    flows[(1, 0, 0)] = [imperfect_flow_10, imperfect_flow_10_level1]
    _, _, _, not_occluded_masks, _, _ = \
        uflow_utils.compute_warps_and_occlusion(
            flows, occlusion_estimation='brox')
    # assert that everything is occluded
    is_zeros_01 = np.equal(
        np.zeros((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(0, 1, 0)][0][:, 4:-4, 4:-4, :]).all()
    is_zeros_10 = np.equal(
        np.zeros((batch_size, height - 8, width - 8, 1)),
        not_occluded_masks[(1, 0, 0)][0][:, 4:-4, 4:-4, :]).all()
    self.assertTrue(is_zeros_01)
    self.assertTrue(is_zeros_10)

  def test_resize_sparse_flow(self):
    flow = torch.tensor(
        [[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.float32)
    mask = torch.tensor([[[1], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0]]],
                       dtype=torch.float32)
    flow_result = torch.tensor([[[0.25, 0], [0, 0]], [[0, 0], [0, 0]]],
                              dtype=torch.float32)
    mask_result = torch.tensor([[[1], [0]], [[0], [0]]], dtype=torch.float32)
    flow_resized, mask_resized = uflow_utils.resize(
        flow, 2, 2, is_flow=True)
    # flow_resized, mask_resized = uflow_utils.resize(
    #     flow, 2, 2, is_flow=True, mask=mask)
    flow_okay = torch.equal(flow_resized, flow_result)
    # mask_okay = torch.equal(mask_resized, mask_result)
    self.assertTrue(flow_okay)
    # self.assertTrue(mask_okay)


if __name__ == '__main__':
  absltest.main()
