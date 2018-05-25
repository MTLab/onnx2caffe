import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.onnx as onnx
import os
import numpy as np
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
class broadcast_mul(nn.Module):
    def __init__(self):
        super(broadcast_mul, self).__init__()
        self.conv1 = conv_bn(3,128,1)
        self.poo1 = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.poo1(x1)
        # x2 = x2.view(x2.size(0), x2.size(1))
        out = x1*x2
        return out

def export(dir):
    dummy_input = Variable(torch.randn(1, 3, 4, 4))
    model = broadcast_mul()
    model.eval()
    torch.save(model.state_dict(),os.path.join(dir,"broadcast_mul.pth"))
    onnx.export(model, dummy_input,os.path.join(dir,"broadcast_mul.onnx"), verbose=True)


def get_model_and_input(model_save_dir):
    model = broadcast_mul()
    model.cpu()
    model_path = os.path.join(model_save_dir,'broadcast_mul.pth')
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    batch_size = 1
    channels = 3
    height = 4
    width = 4
    images = Variable(torch.ones(batch_size, channels, height, width))
    return images,model
