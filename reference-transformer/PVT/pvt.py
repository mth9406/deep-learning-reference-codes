from importlib.resources import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint

class Mlp(nn.Module):

    def __init__(self, 
                 in_features: int, 
                 hidden_features: None,
                 out_features: None,
                 act_layer= nn.GELU,
                 drop= 0.
                ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    
    def __init__(self, dim, num_heads= 8, qkv_bias= False, qk_scale= None, 
                attn_drop= 0., proj_drop= 0., sr_ratio= 1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        

        self.q = nn.Linear(dim, dim, bias= qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias= qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size= sr_ratio, stride= sr_ratio) 
            # H(or W) <- H(or W)/sr_ratio
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # B HW C(dim)
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        # B, num_heads, N, C//num_heads 
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # HW/R**2
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1 ,4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1 ,4)
            # B, N, (2C) -> B, N, 2, num_heads, C//num_heads -> 2, B, num_heads, N, C//num_heads
        k, v = kv[0], kv[1]

        attn = F.softmax(q@k.transpose(-2,-1)*self.scale, dim= -1)
        attn = self.attn_drop(attn) # B, num_heads, N, N

        x = attn@v # B, num_heads, N, C//num_heads
        x = x.transpose(2,1).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 mlp_ratio:int = 4,
                 qkv_bias:bool = False,
                 qk_scale = None,
                 proj_drop = 0.,
                 attn_drop= 0.,
                 drop_path= 0.,
                 act_layer= nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio= 1
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads,
            qkv_bias, qk_scale,
            attn_drop, proj_drop, sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # if drop_path > 0
        # deactivate some weights in a block.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features= mlp_hidden_dim, 
                        act_layer= act_layer, drop= proj_drop)
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    '''Image to patch embedding
    '''
    def __init__(self, img_size= 224, patch_size= 16, in_channels= 3, emb_size= 768):
        super().__init__()
        img_size = to_2tuple(img_size) # (img_size, img_size)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size       
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size= patch_size, stride= patch_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1,2) 
        # B, C, H, W -> B emb_size, H/P W/P 
        # -> B emb_size HW/P^2 -> B HW/P^2 emb_size
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        
        return x, (H, W)

class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths # the number of encoder blocks per transformer layer
        self.F4 = F4
        self.num_stages = num_stages # the number of transformer layers

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages): # 0, 1, 2, 3
            
            # patch embedding of i'th layer
            patch_embed = PatchEmbed(img_size if i == 0 else img_size//2**(i+1), 
                                    patch_size=patch_size if i == 0 else 2, 
                                    # img_size, img_size // 4, img_size // 8, img_size // 16 , img_size//32
                                    in_chans= in_chans if i == 0 else embed_dims[i-1],
                                    embed_dims= embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages -1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeroes(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(drop_rate)

            # blocks of i'th layer
            block = nn.ModuleList(
                [Block(
                    embed_dims[i], num_heads[i], mlp_ratios[i], 
                    qkv_bias = qkv_bias, qk_scale= qk_scale, drop= drop_rate, 
                    attn_drop= attn_drop_rate, drop_path = dpr[cur+j],
                    norm_layer= norm_layer, sr_ratio= sr_ratios[i]
                ) for j in range(depths[i])]
            ) 
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std= 0.02)

        # init weights
        self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        return x
