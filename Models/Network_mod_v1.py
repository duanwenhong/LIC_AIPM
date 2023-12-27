import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from .FullFactorizedModel import FullFactorizedModel
from .ConditionalGaussianModel import ConditionalGaussianModel
from .MaskedConv2d import MaskedConv2d
from .GMM import Distribution_for_entropy2
from .quantize import quantize
from .ResUnit import EncResUnit, DecResUnit


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5), "kernel size must be 3 or 5"
        padding = 2 if kernel_size == 5 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inputs = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        att = inputs * self.sigmoid(x)
        return inputs + att

class SpatialAttention_simple(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention_simple, self).__init__()
        assert kernel_size in (3, 5), "kernel size must be 3 or 5"
        padding = 2 if kernel_size == 5 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inputs = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        att = inputs * self.sigmoid(x)
        return att



class channel_attention_layer(nn.Module):
    def __init__(self, reduction=16):
        super(channel_attention_layer, self).__init__()
        self.channel = 192
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(self.channel, self.channel//reduction, 1, padding = 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel // reduction, self.channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y
# channel attention block
class RCAB(nn.Module):
    def __init__(self):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.PReLU(192),
            nn.Conv2d(192, 192, 3, 1, 1),
            channel_attention_layer()
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)

    def forward(self, x):
        x1 = self.conv2(F.relu(self.conv1(x)))
        out = x + x1
        return out

class ResBlock_new(nn.Module):
    def __init__(self, channels, features):
        super().__init__()
        self._c = channels
        self._f = features

        self.conv1 = nn.Conv2d(192, 96, 1, 1, 0, padding_mode="replicate")
        self.relu = nn.PReLU(96)
        self.conv2 = nn.Conv2d(96, 96, 3, 1, 1, padding_mode="replicate")
        self.relu2 = nn.PReLU(96)
        self.conv3 = nn.Conv2d(96, 192, 1, 1, 0, padding_mode="replicate")

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.conv3(y)

        return y + x

class Non_local_Attention_Block(nn.Module):
    def __init__(self):
        super(Non_local_Attention_Block, self).__init__()
        self.trunk1 = ResBlock_new(192, 192)
        self.trunk2 = ResBlock_new(192, 192)
        self.trunk3 = ResBlock_new(192, 192)
        self.attention1 = ResBlock_new(192, 192)
        self.attention2 = ResBlock_new(192, 192)
        self.attention3 = ResBlock_new(192, 192)
        self.conv1 = nn.Conv2d(192, 192, 1, 1, 0)
    def forward(self, x):
        attention_branch = self.attention1(x)
        attention_branch = self.attention2(attention_branch)
        attention_branch = self.attention3(attention_branch)
        attention_branch = self.conv1(attention_branch)
        attention_branch = F.sigmoid(attention_branch)

        trunk = self.trunk1(x)
        trunk = self.trunk2(trunk)
        trunk = self.trunk3(trunk)
        trunk = x + trunk*attention_branch
        return trunk

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model_Y = nn.Sequential(
            nn.Conv2d(1, self._nlc, 5, 2, 2),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1),
            RCAB(),
            SpatialAttention_simple(),
            Non_local_Attention_Block()
        )

        # self._shortcut_Y = nn.Sequential(
        #     nn.Conv2d(1, self._nlc, 1, 2)
        # )

        self._model_UV = nn.Sequential(
            nn.Conv2d(2, self._nlc, 3, 1, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1),
            RCAB(),
            SpatialAttention_simple(),
            Non_local_Attention_Block()
        )
        
        self.intergrate = nn.Sequential(
            nn.Conv2d(2*self._nlc, self._nlc, 1, 1, 0),
            nn.PReLU(self._nlc)
        )

        self.trunk2 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.trunk3 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 5, 2, 2),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 1, 2)
        )

        self.attention1_channel = channel_attention_layer()
        self.attention1 = Non_local_Attention_Block()
        # self.mask1 = nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))
        # self.attention1_trunk =  nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))
        
        self.trunk4 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.trunk5 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 5, 2, 2),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut5 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 1, 2)
        )

        self.trunk6 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.trunk7 = nn.Conv2d(self._nlc, self._noc, 5, 2, 2)
        self.attention2_channel = channel_attention_layer()
        self.attention2 = Non_local_Attention_Block()
        # self.mask2 = nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))
        # self.attention2_trunk =  nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))

    def forward(self, inputs_Y, inputs_UV):
        Y_out = self._model_Y(inputs_Y)
        UV_out = self._model_UV(inputs_UV)
        out = torch.cat([Y_out, UV_out], dim=1)
        out = self.intergrate(out)
        out_2 = self.trunk2(out) + out
        out_3 = self.trunk3(out_2) + self.shortcut3(out_2)
        out_3_att1 = self.attention1_channel(out_3) 
        out_3_att2 = self.attention1(out_3_att1)
        # out_3_att2 = self.attention1_trunk(out_3_att1)*F.sigmoid(self.mask1(out_3_att1)) + out_3_att1
        out_4 = self.trunk4(out_3_att2) + out_3_att2
        out_5 = self.trunk5(out_4) + self.shortcut5(out_4)
        out_6 = self.trunk6(out_5) + out_5
        out_7 = self.trunk7(out_6)
        out_7_att1 = self.attention2_channel(out_7)
        out_7_att2 = self.attention2(out_7_att1)
        # out_7_att2 = self.attention2_trunk(out_7_att1)*F.sigmoid(self.mask2(out_7_att1)) + out_7_att1
        return out_7_att2


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels
        #modify 这个地方原来最后一个DecRes对应的noc 应该是nlc

        self.attention1_channel = channel_attention_layer()
        self.attention1 = Non_local_Attention_Block()
        # self.mask1 = nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))
        # self.attention1_trunk = nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))
        
        self.trunk1 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )
        
        self.trunk2 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 5, 2, 2, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )
        self.shortcut2 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 1, 2, 0, 1)
        )

        self.trunk3 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.trunk4 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 5, 2, 2, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut4 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 1, 2, 0, 1)
        )

        self.attention2_channel = channel_attention_layer()
        self.attention2 = Non_local_Attention_Block()
        # self.mask2 = nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))
        # self.attention2_trunk = nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))
        
        self.trunk5 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.trunk6 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 5, 2, 2, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut6 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 1, 2, 0, 1)
        )

        self.trunk7 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1),
            # EncResUnit(self._nlc, self._nlc, 1)
        )

        self.split = nn.Sequential(
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, 2*self._nlc, 1, 1, 0)
        )
    
        self._model_Y = nn.Sequential(
            RCAB(),
            SpatialAttention_simple(),
            Non_local_Attention_Block(),
            nn.ConvTranspose2d(self._nlc, self._nlc, 5, 2, 2, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, 1, 3, 1, 1)
        )

        # self._shortcut_Y = nn.Sequential(
        #     nn.ConvTranspose2d(self._nlc, 1, 1, 2, 0, 1)
        # )
        self._model_UV = nn.Sequential(
            RCAB(),
            SpatialAttention_simple(),
            Non_local_Attention_Block(),
            nn.ConvTranspose2d(self._nlc, self._nlc, 3, 1, 1),
            nn.PReLU(self._nlc),
            nn.ConvTranspose2d(self._nlc, 2, 3, 1, 1)
        )

    def forward(self, inputs):
        out_att1 = self.attention1_channel(inputs)
        out_att2 = self.attention1(out_att1)
        # out_att2 = self.attention1_trunk(out_att1)*F.sigmoid(self.mask1(out_att1)) + out_att1
        out_1 = self.trunk1(out_att2) + out_att2
        out_2 = self.trunk2(out_1) + self.shortcut2(out_1)
        out_3 = self.trunk3(out_2) + out_2
        out_4 = self.trunk4(out_3) + self.shortcut4(out_3)
        
        out_4_att1 = self.attention2_channel(out_4)
        out_4_att2 = self.attention2(out_4_att1)
        # out_4_att2 = self.attention2_trunk(out_4_att1)*F.sigmoid(self.mask2(out_4_att1)) + out_4_att1

        out_5 = self.trunk5(out_4_att2) + out_4_att2
        out_6 = self.trunk6(out_5) + self.shortcut6(out_5)
        out_7 = self.split(out_6)
        out_Y = out_7[:, 0:self._nlc, :, :]
        out_UV = out_7[:, self._nlc:2*self._nlc, :, :]
        out_Y = self._model_Y(out_Y)
        out_UV = self._model_UV(out_UV)
        return {
            "output_Y" : out_Y,
            "output_UV" : out_UV
        }

class HyperEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._hyper_encoder = nn.Sequential(
            nn.Conv2d(self._nic, self._nlc, 5, 2, 2),
            EncResUnit(192, 192, 1),
            EncResUnit(192, 192, 1),
            nn.Conv2d(self._nlc, self._noc, 5, 2, 2),
            EncResUnit(192, 192, 1),
            EncResUnit(192, 192, 1),
        )
        self.attention_channel = channel_attention_layer()
        self.attention = Non_local_Attention_Block()
        # self.mask =  nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))
        # self.attention_trunk = nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))

    def forward(self, inputs):
        out_1 = self._hyper_encoder(inputs)
        out_1_att1 = self.attention_channel(out_1)
        out_2 = self.attention(out_1_att1)
        # out_2 = self.attention_trunk(out_1_att1)*F.sigmoid(self.mask(out_1_att1)) + out_1_att1
        return out_2


class HyperDecoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(HyperDecoder, self).__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._hyper_decoder = nn.Sequential(
            EncResUnit(192, 192, 1),
            EncResUnit(192, 192, 1),
            nn.ConvTranspose2d(192, 192, 5, 2, 2, 1),
            EncResUnit(192, 192, 1),
            EncResUnit(192, 192, 1),
            nn.ConvTranspose2d(192, 384, 5, 2, 2, 1)
        )
        
        self.attention_channel = channel_attention_layer()
        self.attention = Non_local_Attention_Block()
        # self.attention_trunk = nn.Sequential(ResBlock(192, 192, 3, 1, 1), ResBlock(192, 192, 3, 1, 1),
        #                             ResBlock(192, 192, 3, 1, 1))
        # self.mask =  nn.Sequential(Non_local_Block(192, 96), ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1),
        #                            ResBlock(192, 192, 3, 1, 1), nn.Conv2d(192, 192, 1, 1, 0))

    def forward(self, inputs):
        out_att1 = self.attention_channel(inputs)
        out_att2 = self.attention(out_att1)
        # out_att2 = self.attention_trunk(out_att1) + F.sigmoid(self.mask(out_att1)) + out_att1
        out_2 = self._hyper_decoder(out_att2)
        return out_2

class EntropyParameters_GMM(nn.Module):
    def __init__(self):
        super(EntropyParameters_GMM, self).__init__()
        # self._ncs = [int(item) for item in np.linspace(in_channels, out_channels, 4)]

        self._entropy_parameters = nn.Sequential(
            nn.Conv2d(768, 640, 1),
            nn.PReLU(640),
            nn.Conv2d(640, 640, 1),
            nn.PReLU(640),
            nn.Conv2d(640, 1728, 1)
        )

    def forward(self, inputs):
        return self._entropy_parameters(inputs)

class Network_v1(nn.Module):
    def __init__(self, channels, context):
        super().__init__()
        self._context = context
        self._nc = channels

        # Define the network
        self.encoder = Encoder(6, self._nc, self._nc)
        self.decoder = Decoder(self._nc, self._nc, 6)
        self.hyper_encoder = HyperEncoder(self._nc, self._nc, self._nc)
        self.hyper_decoder = HyperDecoder(self._nc, self._nc*3//2, self._nc*2)
        if self._context:
            self.context_model = MaskedConv2d(self._nc, self._nc*2, 3, 1, 1)
            # self.entropy_parameters = EntropyParameters(self._nc*4, self._nc*2)
            self.entropy_parameters = EntropyParameters_GMM()
        # Define the entropy model
        self.factorized = FullFactorizedModel(self._nc, (3, 3, 3), 1e-9)
        # self.conditional = ConditionalGaussianModel(1e-3, 1e-9)
        self.GMM = Distribution_for_entropy2() 

    @staticmethod
    def squeeze(x, r):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // r, r, W // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (r ** 2), H // r, W // r)

        return x

    @staticmethod
    def unsqueeze(x, r):
        return F.pixel_shuffle(x, r)

    def forward(self, y_comp, u_comp, v_comp):
        # Get the inputs
        # x = torch.cat([self.squeeze(y_comp, 2), u_comp, v_comp], dim=1)
        #padding the inputs duan modify
        #print(x.shape)
        uv_comp = torch.cat([u_comp, v_comp], dim = 1)

        batch, channels, height, width = uv_comp.shape

        h_pad = height % 32
        w_pad = width % 32
        if h_pad != 0:
            h_new = (height // 32 + 1) * 32
            h_pad = (h_new - height) // 2
        
        if w_pad != 0:
            w_new = (width // 32 + 1) * 32
            w_pad = (w_new - width) // 2
        uv_comp2 = F.pad(uv_comp, pad = (w_pad, w_pad, h_pad, h_pad), mode='replicate')
        y_comp2 = F.pad(y_comp, pad = (2*w_pad, 2*w_pad, 2*h_pad, 2*h_pad), mode='replicate')
        # batch, channels, height_new, width_new = x2.shape
                
        # Get the reconstructed images
        y = self.encoder(y_comp2, uv_comp2)
        y_hat = quantize(y, self.training)
        
        
        x_hat = self.decoder(y_hat)
        
        # Compute the probability
        z = self.hyper_encoder(y)
        z_hat, z_prob = self.factorized(z)
        if self._context:
            u = self.hyper_decoder(z_hat)
            v = self.context_model(y_hat)
            p = self.entropy_parameters(torch.cat((u, v), dim=1))
        else:
            p = self.hyper_decoder(z_hat)
        # loc, scale_minus_one = torch.split(p, self._nc, dim=1)
        # y_prob = self.conditional(y_hat, loc, scale_minus_one)
        y_prob = self.GMM(y_hat, p)
        h_num  = height * 2
        w_num = width * 2
        bpp = (torch.sum(-torch.log2(y_prob)) + torch.sum(-torch.log2(z_prob))) / batch / (h_num * w_num)
        
        # delete the padding duan modify
        # if h_pad != 0:
        #     x_hat = x_hat[:, :, h_pad:-h_pad, :]
        # if w_pad != 0:
        #     x_hat = x_hat[:, :, :, w_pad:-w_pad]
        
        # Get the reconstruction
        y_comp_hat = x_hat["output_Y"]
        uv_comp_hat = x_hat["output_UV"]

        if h_pad != 0:
            y_comp_hat = y_comp_hat[:, :, 2*h_pad:-2*h_pad, :]
            uv_comp_hat = uv_comp_hat[:, :, h_pad:-h_pad, :]
        if w_pad != 0:
            y_comp_hat = y_comp_hat[:, :, :, 2*w_pad:-2*w_pad]
            uv_comp_hat = uv_comp_hat[:, :, :, w_pad:-w_pad]
        # y_comp_hat = self.unsqueeze(x_hat[:, :4, :, :], 2)
        
        # u_comp_hat = torch.unsqueeze(x_hat[:, 4, :, :], dim=1)
        # v_comp_hat = torch.unsqueeze(x_hat[:, 5, :, :], dim=1)
        
        return {
            "Y":y_comp_hat,
            "UV":uv_comp_hat,
            "bpp":bpp
        }
    #     return {
    #     "outputs": x_hat,
    #     "Y": y_comp_hat, "U": u_comp_hat, "V": v_comp_hat,
    #     "codes": y_hat, "prior": z_hat,
    #     "bpp": bpp, "P_codes": y_prob, "P_prior": z_prob,
    # }

    @property
    def integer_offset_error(self):
        return self.factorized.integer_offset_error()
