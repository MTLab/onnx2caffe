import os
import torch
import argparse
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.onnx as onnx


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.featuresxxx = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )


        self.featuresa = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=4, bias=True),
# nn.Conv2d(3, 2, kernel_size=11, stride=4, padding=2),
#            nn.ReLU(inplace=True),
#nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        return x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
        #return nn.functional.log_softmax(x)


    def alexnet(pretrained=False, **kwargs):
        r"""AlexNet model architecture from the
        `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

        Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
        model = AlexNet(**kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        return model


def export(dir):
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    model = AlexNet()
    # model = load_network(model,os.path.join(file_dir,'..','model','pose_v02.pth'))
    model.eval()
    torch.save(model.state_dict(),os.path.join(dir,"alexnet.pth"))
    onnx.export(model, dummy_input,os.path.join(dir,"alexnet.onnx"), verbose=True)

def get_model_and_input(model_save_dir):
    model = AlexNet()
    model.cpu()
    model_path = os.path.join(model_save_dir,'alexnet.pth')
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    images = Variable(torch.ones(batch_size, channels, height, width))
    return images,model



