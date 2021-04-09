import torch
import torch.nn.functional as F
import numpy as np

flow = torch.arange(0,150,1,dtype= torch.float32).reshape((3,5,5,2))
print(flow.shape)
# flow = torch.ones((3,172,224,2))
# # flow = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
# #                       [[7., 8.], [9., 10.], [11., 12.]]]])
#

flow = flow[None]
m = torch.nn.ReplicationPad3d((0,0,1,1,1,1))
flow = m(flow)
flow = torch.squeeze(flow,0)
# flow = F.pad(flow, pad= (0,0,1,1), mode= "replicate")
# print(flow[:,0,:,:])
