import torch
import torch.nn.functional as F

flow = torch.ones((1,172,224,2))
flow = F.pad(input= flow, pad= [[0, 0], [1, 1], [1, 1], [0, 0]],mode='SYMMETRIC')