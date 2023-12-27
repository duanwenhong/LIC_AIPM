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
            nn.Conv2d(1, self._nlc, 3, 1, 1),
            nn.PReLU(self._nlc),
            
        )
        self._Y_attention = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 5, 2, 2),
            RCAB(),
            Non_local_Attention_Block()
        )

        self._model_UV = nn.Sequential(
            nn.Conv2d(2, self._nlc, 3, 1, 1),
            nn.PReLU(self._nlc),
        )

        self._UV_attention = nn.Sequential(
            RCAB(),
            Non_local_Attention_Block()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.intergrate = nn.Sequential(
            nn.Conv2d(2*self._nlc, self._nlc, 1, 1, 0),
            nn.PReLU(self._nlc)
        )

        self.trunk2 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
        )

        self.trunk3 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 5, 2, 2),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(self._nlc, self._nlc, 1, 2)
        )

        self.attention1_channel = RCAB()
        self.attention1 = Non_local_Attention_Block()
        self.trunk4 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
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
        )

        self.trunk7 = nn.Conv2d(self._nlc, self._noc, 5, 2, 2)
        self.attention2_channel = RCAB()
        self.attention2 = Non_local_Attention_Block()

    def squeeze(self, x, r):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // r, r, W // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (r ** 2), H // r, W // r)
        x = x.reshape(B, C, r ** 2, H // r, W // r)
        x = x.permute(2, 0, 1, 3, 4)
        # print(x.shape)
        return x

    @staticmethod
    def unsqueeze(x, r):
        return F.pixel_shuffle(x, r)
# modify for dual-guide
    def forward(self, inputs_Y, inputs_UV):
        Y_out = self._model_Y(inputs_Y)
        Y_guide = self.squeeze(Y_out, 2)
        Y_split = self.squeeze(inputs_Y, 2)#[4,B,C,H//2,W//2]
        UV_out = self._model_UV(inputs_UV)
        U_split = inputs_UV[:,0:1,:,:]
        V_split = inputs_UV[:,1:2,:,:]#[B,C,H//2,W//2]
        att_score_UV = Y_split * U_split + Y_split * V_split
        att_score_UV = torch.sum(att_score_UV, dim = (2, 3, 4))#[4, B]
        att_para = torch.softmax(att_score_UV, dim = 0)#[4, B]
        [T, B] = list(att_para.size())
        att_para = att_para.reshape(T, B, 1, 1, 1)
        # print(att_para.shape)
        YUV_guide = Y_guide * att_para
        YUV_guide = torch.sum(YUV_guide, dim = 0)
        #remove the Y and replace by Y + UV_feature_transfer
        # Y_feature = self._Y_attention(Y_out)
        # if exist gate control
        # gate_para = self.gate(UV_out)
        # UV_feature = gate_para * UV_out + (1 - gate_para) * YUV_guide
        UV_feature = UV_out + YUV_guide
        UV_feature_transfer = torch.unsqueeze(UV_out, 0)
        _, B_, C_, H_, W_ = UV_feature_transfer.size()

        UV_feature_transfer = UV_feature_transfer.expand(4, B_, C_, H_, W_)
        #2022.5.21 notice lacking the next one python line to achieve the attention map
        UV_feature_transfer = UV_feature_transfer * att_para
        UV_feature_transfer = UV_feature_transfer.permute(1, 2, 0, 3, 4)
        UV_feature_transfer = UV_feature_transfer.reshape(B_,C_ * 4, H_, W_)
        UV_feature_transfer = self.unsqueeze(UV_feature_transfer, 2)
        # print(UV_feature_transfer.shape)
        # break
        Y_feature = Y_out + UV_feature_transfer
        Y_feature = self._Y_attention(Y_feature)

        UV_feature = self._UV_attention(UV_feature)


        # att_score_V = Y_split * V_split
        # att_score_U = torch.sum(att_score_U, dim=(2, 3, 4))
        # att_score_V = torch.sum(att_score_V, dim=(2, 3, 4))


        # UV_out = self._model_UV(inputs_UV)
        out = torch.cat([Y_feature, UV_feature], dim=1)
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

        self.attention1_channel = RCAB()
        self.attention1 = Non_local_Attention_Block()

        self.trunk1 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
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
        )

        self.trunk4 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 5, 2, 2, 1),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1)
        )

        self.shortcut4 = nn.Sequential(
            nn.ConvTranspose2d(self._nlc, self._nlc, 1, 2, 0, 1)
        )

        self.attention2_channel = RCAB()
        self.attention2 = Non_local_Attention_Block()
        
        self.trunk5 = nn.Sequential(
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
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
        )

        self.split = nn.Sequential(
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, 2*self._nlc, 1, 1, 0)
        )
    
        self._model_Y = nn.Sequential(
            RCAB(),
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
        self.attention_channel = RCAB()
        self.attention = Non_local_Attention_Block()
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
        
        self.attention_channel = RCAB()
        self.attention = Non_local_Attention_Block()

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

class Network_dual_guide(nn.Module):
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
        
        # Get the reconstruction
        y_comp_hat = x_hat["output_Y"]
        uv_comp_hat = x_hat["output_UV"]

        if h_pad != 0:
            y_comp_hat = y_comp_hat[:, :, 2*h_pad:-2*h_pad, :]
            uv_comp_hat = uv_comp_hat[:, :, h_pad:-h_pad, :]
        if w_pad != 0:
            y_comp_hat = y_comp_hat[:, :, :, 2*w_pad:-2*w_pad]
            uv_comp_hat = uv_comp_hat[:, :, :, w_pad:-w_pad]
        
        return {
            "Y":y_comp_hat,
            "UV":uv_comp_hat,
            "bpp":bpp
        }

    @property
    def integer_offset_error(self):
        return self.factorized.integer_offset_error()
