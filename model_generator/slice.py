import torch
import torch.nn as nn
import os


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
        self.conv = nn.Conv2d(3, 4, 3, 1, 1, dilation=1)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :2, ...]


def export(dir):
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)
    dummy_input = torch.randn(1, 3, 32, 32)
    model = Slice()
    # model = load_network(model,os.path.join(file_dir,'..','model','pose_v02.pth'))
    model.eval()
    torch.save(model.state_dict(),os.path.join(dir,"slice.pth"))
    torch.onnx.export(model, dummy_input,os.path.join(dir,"slice.onnx"), verbose=True)


def get_model_and_input(model_save_dir=None):
    model = Slice()
    model.cpu()
    if model_save_dir is not None:
        model_path = os.path.join(model_save_dir, 'slice.pth')
        model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    batch_size = 1
    channels = 3
    height = 32
    width = 32
    images = torch.ones(batch_size, channels, height, width)
    return images, model