import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import tensorflow as tf

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter()
# writer.set_as_default()
#
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# model = torchvision.models.resnet50(False)
# # Have ResNet model take in grayscale rather than RGB
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# images, labels = next(iter(trainloader))
#
# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
# writer.close()
# flow = torch.arange((1*2*3*3), dtype= torch.float32).reshape(1,2,3,3)
# #
# # flow = img[None]
# m = torch.nn.ReplicationPad2d(( 1, 1, 1, 1))
# flow = m(flow)
# print(flow)
flow = tf.reshape(tf.range(1*2*3*3, dtype = tf.float32),(1,3,3,2))
# print(flow)
flow = tf.pad(
            tensor=flow,
            paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC')

print(flow)