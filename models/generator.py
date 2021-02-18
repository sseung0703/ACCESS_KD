import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        bn_eps = 1e1
        
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8 ** 2),
            nn.LeakyReLU(0.2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, 3, stride=1, padding=1),

            nn.BatchNorm2d(3, eps=bn_eps)
        )

        self._init_weight()

    def _init_weight(self):
        modules = self.layers.modules()
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)


if __name__ == '__main__':
    from torchsummary import summary

    z = torch.randn((50,100), dtype=torch.float)
    model = Generator(z_dim=100)

    model.print_shape(z)
    images = model(z)
    print(images.shape)

    summary(model, input_size=((1, 100)))
