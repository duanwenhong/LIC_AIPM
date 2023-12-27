from numpy.core.numeric import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, nic, nlc):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nic, nlc, 3, padding=1)
        self.conv2 = nn.Conv2d(nlc, nic, 3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = nn.functional.relu(y, inplace=True)
        y = self.conv2(y)

        return x + y


class LoopFilter_U(nn.Module):
    def __init__(self, channels):
        super(LoopFilter_U, self).__init__()
        self._ch = channels
        # duan modify 3 channel to 6 channel input
        self.conv1_part1 = nn.Conv2d(1, self._ch//2, 3, padding=1)
        self.conv1_part2 = nn.Conv2d(1, self._ch//2, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(self._ch, self._ch) for _ in range(20)])
        self.conv2 = nn.Conv2d(self._ch, self._ch, 3, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(self._ch, 1, 3, padding=1),
                                #    nn.AvgPool2d(2, 2),
                                   nn.Conv2d(1, 1, 5, 2, 2)
                                )

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=1e-3)
                    nn.init.zeros_(m.bias)

    def forward(self, inputs_y, inputs_u):
        identity = inputs_u
        y = self.conv1_part1(inputs_y)
        y = nn.functional.relu(y, inplace=True)
        u = F.interpolate(inputs_u, scale_factor=2, mode="bicubic")
        # print(u.shape)
        # print(inputs_u.shape)
        u = self.conv1_part2(u)
        u = nn.functional.relu(u, inplace=True)
        out = torch.cat([y, u], 1)
        # y = nn.functional.relu(y, inplace=True)

        z = self.blocks(out)

        w = self.conv2(z)
        w = nn.functional.relu(w, inplace=True)
        w = self.conv3(w)
        return w + identity

# class LoopFilter_V(nn.Module):
#     def __init__(self, channels):
#         super(LoopFilter_V, self).__init__()
#         self._ch = channels
#         # duan modify 3 channel to 6 channel input
#         self.conv1_part1 = nn.Conv2d(1, self._ch//2, 3, padding=1)
#         self.conv1_part2 = nn.Conv2d(1, self._ch//2, 3, padding=1)
#         self.blocks = nn.Sequential(*[ResBlock(self._ch, self._ch) for _ in range(10)])
#         self.conv2 = nn.Conv2d(self._ch, self._ch, 3, padding=1)
#         self.conv3 = nn.Sequential(nn.Conv2d(self._ch, 1, 3, padding=1),
#                                    nn.AvgPool2d(2, 2),
#                                 )

#         self.reset_parameters()

#     def reset_parameters(self):
#         with torch.no_grad():
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.normal_(m.weight, std=1e-3)
#                     nn.init.zeros_(m.bias)

#     def forward(self, inputs_y, inputs_v):
#         identity = inputs_v
#         y = self.conv1_part1(inputs_y)
#         y = nn.functional.relu(y, inplace=True)
#         v = F.interpolate(inputs_v, scale_factor=2, mode="nearest")
#         v = self.conv1_part2(v)
#         v = nn.functional.relu(v, inplace=True)
#         out = torch.cat([y, v], 1)
#         # y = nn.functional.relu(y, inplace=True)

#         z = self.blocks(out)

#         w = self.conv2(z)
#         w = nn.functional.relu(w, inplace=True)
#         w = self.conv3(w)
#         return w + identity

class LoopFilter_UV(nn.Module):
    def __init__(self, channels):
        super(LoopFilter_UV, self).__init__()
        self._ch = channels
        # duan modify 3 channel to 6 channel input
        self.conv1 = nn.Conv2d(6, self._ch, 3, padding=1)
        
        self.blocks = nn.Sequential(*[ResBlock(self._ch, self._ch) for _ in range(20)])
        self.conv2 = nn.Conv2d(self._ch, self._ch, 3, padding=1)
        self.conv3 = nn.Conv2d(self._ch, 6, 3, padding=1)
                                

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=1e-3)
                    nn.init.zeros_(m.bias)

    def forward(self, inputs_uv):
        x = inputs_uv
        y = self.conv1(x)
        y = nn.functional.relu(y, inplace=True)

        z = self.blocks(y)

        w = self.conv2(z)
        w = nn.functional.relu(w, inplace=True)
        w = self.conv3(w)

        return x + w

if __name__ == "__main__":
    dtypes = {"dtype": torch.float32, "device": torch.device("cuda:0")}
    net = LoopFilter_UV(64).to(**dtypes)
    
    x = torch.zeros(1, 3, 128, 128).normal_(0, 0.3).to(**dtypes)
    z = net(x)

    net.zero_grad()
    z.backward(torch.ones_like(z))

    print(z.shape)

