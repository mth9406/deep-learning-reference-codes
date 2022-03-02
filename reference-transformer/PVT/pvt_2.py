import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class Mlp(nn.Module):
    
    def __init__(self, 
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 act_layer = nn.GELU,
                 drop= 0.
                ):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        super().__init__()
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

class SpatialReductionAttention(nn.Module):
    '''Spatial reduction attention,
       ordinary attention block when sr_ratio = 1
    '''
    def __init__(self, dim, num_heads= 8, 
                 qkv_bias = False, qk_scale= None,
                 attn_drop= 0., proj_drop= 0., sr_ratio= 1
                ):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim: {dim} should be divisible by num_heads: {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        # embedding is equally distributed to each head.
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias= qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias= qkv_bias)
        
        self.proj = nn.Linear(dim, dim) # project attention@value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape # B HW C(dim)
        
        # create query vectors
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        # B, N, C -> B, N, dim -> B, num_heads, N, dim//num_heads

        # create key, value vectors
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C//self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4)
            # B, N, C -> B, N, dim*2 
            # -> B, N, num_heads, dim*2//num_heads
            # -> B, N, num_heads, 2 dim//num_heads
            # -> 2, B, num_heads, N, dim//num_heads
        k, v = kv[0], kv[1]
        # B, num_heads, N, dim//num_heads
        
        attn = F.softmax(q@k.transpose(-1,-2)*self.scale, dim= -1)
        # B, num_heads, N, N
        attn= self.attn_drop(attn)
        
        x = attn@v
        # B, num_heads, N, dim//num_heads
        x = x.permute(0, 2, 1, 3).reshape(B, N, C) # B, N, C
        x = self.proj(x) # B, N, C
        x = self.proj_drop(x)

        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self, 
                 dim: int, 
                 num_heads: int,
                 mlp_ratio: int = 4, 
                 qkv_bias: bool = False, 
                 qk_scale = None, 
                 proj_drop: float = 0.,
                 attn_drop: float = 0., 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm, 
                 sr_ratio: int = 1
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(
            dim= dim,
            num_heads= num_heads,
            qkv_bias= qkv_bias,
            qk_scale= qk_scale,
            attn_drop= attn_drop,
            proj_drop= proj_drop,
            sr_ratio= sr_ratio
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features= dim, hidden_features= mlp_hidden_dim,
                        act_layer= act_layer, drop= proj_drop)
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    '''Image to patch embedding
    '''
    def __init__(self, 
                 img_size:int = 224, 
                 patch_size:int = 16, 
                 in_chans:int = 3,
                 dim:int = 768
                ):
        super().__init__()
        img_size = to_2tuple(img_size) # (img_size, img_size)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, dim, 
                              kernel_size= patch_size, 
                              stride= patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(-1,-2)
        # B, C, H, W -> B, dim, H//P, W//P
        # -> B, dim HW//P^2 -> B, HW//P^2 dim
        x = self.norm(x)
        H, W = H//self.patch_size[0], W//self.patch_size[1]

        return x, (H, W)

patchemb = PatchEmbed(dim = 64)
sra = SpatialReductionAttention(dim = 64, sr_ratio= 2)

x = torch.randn(1, 3, 224, 224)
patches, hw = patchemb(x)
out = sra(patches, *hw)

print(x.shape)
print(patches.shape)
print(out.shape)
        