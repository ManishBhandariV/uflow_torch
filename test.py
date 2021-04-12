import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
# import tensorflow as tf

# flow = torch.arange(0,77056,1,dtype= torch.float32).reshape((172,224,2))
# flow
# img = np.load("/home/manish/Desktop/000000_10.npy")

# img = np.moveaxis(img,-1, 0)
# print(img.shape)

# print(flow.shape)
# # flow = torch.ones((3,172,224,2))
# # # flow = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
# # #                       [[7., 8.], [9., 10.], [11., 12.]]]])
# #
#
# # flow = flow[None]
# # m = torch.nn.ReplicationPad3d((0,0,1,1,1,1))
# # flow = m(flow)
# # flow = torch.squeeze(flow,0)
# # flow = F.pad(flow, pad= (0,0,1,1), mode= "replicate")
# # print(flow[:,0,:,:])
#
# a = transforms.Compose([transforms.Resize((600,600))])
# img = torch.Tensor(img[None])
# img = torch.moveaxis(img,-1,1)
#
# img = a(img)
# print(img.shape)
# img = torch.moveaxis(img,1,-1)
# plt.imshow(flow)
# plt.show()
img = torch.arange(77056).reshape((1,172,224,2))
height =600
width = 600
orig_height = img.shape[1]

orig_width = img.shape[2]


img = torch.arange(77056).reshape((1,1,172,224,2))
shape = list(img.shape)
img_flattened = torch.reshape(img, [-1] + shape[-3:])
img_flattened.set_shape([-1] + shape[-3:])
