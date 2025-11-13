import torch
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

class VQModelTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h
    
    def encode_list(self, x):
        z_list = []
        start_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        for g in range(26):
           sta_ind = start_idx[g]
           end_ind = sta_ind + 3   
           xi = x[:, sta_ind:end_ind, :, :]
           h = self.encoder(xi)
           h = self.quant_conv(h)
           z_list.append(h)
        return z_list
    
    def decode_list(self, z_list):
        b, c, h, w = z_list[0].shape
        channel_counter = torch.zeros(28).to('cuda:4')
        y = torch.zeros(b, 28, 256, 256).to('cuda:4')
        start_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        for g in range(26):
            sta_ind = start_idx[g]
            end_ind = sta_ind + 3
            z = self.post_quant_conv(z_list[g].half())
            deci = self.decoder(z)
            y[:, sta_ind:end_ind, :, :] += deci
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = ((y / channel_counter.unsqueeze(1).unsqueeze(2)) + 1)  / 2
        return y

    def decode(self, h, force_not_quantize=False, grad_forward=False):
        if not force_not_quantize:
            if grad_forward:
                with torch.no_grad():
                    quant, emb_loss, info = self.quantize(h)
                quant = (quant - h).detach()+h
            else:
                quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant.half())
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec
    

    def forward_list(self, input, force_not_quantize=False):
        z_list = self.encode_list(input)
        dec = self.decode_list(z_list, input, force_not_quantize)
        return dec

class AutoencoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior, return_moments=False)
        dec = self.decode(z)
        return dec

class EncoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z
    def forward(self, x, sample_posterior=True, return_moments=False):
        return self.encode(x, sample_posterior, return_moments)

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

