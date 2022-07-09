# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from radon_transformation import *
import numpy as np

from decoder import *


class dualViT(nn.Module):
    def __init():
       super().__init__()
       
       self.decoder = Decoder(z_channels = 256,resolution=256, in_channels=3, out_ch = 1, ch = 128, ch_mult=[1,1,2,2,4],num_res_blocks = 2,attn_resolutions=[16])
       self.spatial_ViT = ViT(
          patch_size=16, embed_dim=768, depth=4, num_heads=8,
          decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
          mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
       self.motion_ViT = ViT(
          patch_size=16, embed_dim=768, depth=4, num_heads=8,
          decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
          mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial=False)
    
    def forward(self, sinograms, thetas)
       spatial_feature = self.spaital_ViT(sinograms,thetas)
       #spatial_feature = unpatchify(spatial_feature)
       print(spatial_feature.shape)
       motion_feature = self.motion_ViT(sinograms,thetas)
       print(motion_feature.shape)
       #for i in range(thetas.shape[1])
           #feature = motion+spatial

class ViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, spatial = True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.reconstruction_size = 512
        self.angles = 64
        
        self.projection_embed = nn.Linear(512,embed_dim)
        self.embed_dim = embed_dim


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_dim = decoder_embed_dim


        

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 256, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):

        if self.spatial:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 256, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 16, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.projection_embed.weight.data
        torch.nn.init.xavier_uniform_(w)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = 16
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs


    def radon(self, images, thetas):
        bsz, _, shape_size, _ = images.shape
        _, n_angles = thetas.shape
        thetas = thetas.unsqueeze(-1).unsqueeze(-1)
        cos_al, sin_al = thetas.cos(), thetas.sin()
        zeros = torch.zeros_like(cos_al).cuda()
        rotations = torch.stack((cos_al,sin_al,zeros,-sin_al, cos_al,zeros),-1).reshape(-1,2,3)
        rotated = torch.nn.functional.affine_grid(rotations, torch.Size([n_angles*bsz, 1, self.img_size, self.img_size]), align_corners=True).reshape(bsz,-1,self.img_size,2)
        out_fl = torch.nn.functional.grid_sample(images, rotated, align_corners=True).reshape(bsz,1,n_angles,shape_size,shape_size)
        out = out_fl.sum(3)
        return out

    def make_pos_embed(self, embed_dim, thetas):
        
        emb_sin = torch.sin(thetas)
        emb_cos = torch.cos(thetas)
        emb = torch.stack([emb_sin,emb_cos],dim=2).repeat(1,1,embed_dim//2)
        return emb

    def forward_encoder(self, x, thetas):
        # embed patches
        x = self.projection_embed(x)
        # add pos embed w/o cls token
        pos_embed = self.make_pos_embed(self.embed_dim, thetas)
        x = x + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, thetas):
        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed
        if self.spatial:
            x = x + self.decoder_pos_embed
        else:
            pos_embed = self.make_pos_embed(self.embed_dim, thetas)
            x = x + pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        x = torch.nn.functional.sigmoid(x)
        
        return x

    def forward_loss(self, sinograms, reconstruction, thetas, mask):
    
        reconstruction = torch.nn.functional.interpolate(reconstruction, self.reconstruction_size, mode="bilinear")*255
        pred_sinograms = self.radon(reconstruction, thetas)/65535
        loss = torch.nn.functional.mse_loss(pred_sinograms,sinograms)
        
        return loss

    def forward(self, sinograms, thetas):
        latent = self.forward_encoder(sinograms, thetas)
        pred = self.forward_decoder(latent, thetas)  # [N, L, p*p*3]
        #reconstruction = self.unpatchify(pred)
        #loss = self.forward_loss(sinograms,reconstruction, thetas, mask)
        return pred#reconstruction, loss, mask





def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=4, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
