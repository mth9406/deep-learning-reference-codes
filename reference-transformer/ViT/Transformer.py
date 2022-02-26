import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

img = Image.open('./cat.bmp')

# fig = plt.figure()
# plt.imshow(img)

transform = Compose([Resize((224,224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # b, c, h, w

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels: int= 3,
                 patch_size: int= 16, emb_size:int= 768,
                 img_size: int= 224
                ):
        self.patch_size= patch_size
        self.img_size= img_size
        super().__init__()
        # projection of patches
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, patch_size),
            Rearrange('b e h w-> b (h w) e') # flatten the patches...
        )
        # cls token to be added in the begining of each patch 
        self.cls_token = nn.Parameter(
            torch.randn(1,1,emb_size)
        )
        # position embedding
        n_patches = img_size // patch_size
        self.positions = nn.Parameter(
            torch.randn(n_patches**2+1, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x) # b (h w) e
        b, _, _ = x.shape
        # pad cls_tokens
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b= b)
        x = torch.cat([cls_tokens, x], dim= 1)
        # add position embeddings
        # broad casting is applied...
        x += self.positions
        return x

# print(PatchEmbedding()(x).shape)
# ex) batch, num_patches(including cls_tokens), emb

# Transformer
# we can use nn.MultiHeadAttention from PyTorch....
class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size: int= 768, num_heads:int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # to vetorize (generate) key, query, value
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5 # sqrt(d) in the original paper

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:

        # input: (batch, num_patches + clstoken(1), emb_size)
        # generate keys, queries, values from an input
        # obtain heads from embedding dimension...
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h= self.num_heads)
        keys = rearrange(self.queries(x), 'b n (h d) -> b h n d', h= self.num_heads)
        values = rearrange(self.queries(x), 'b n (h d) -> b h n d', h= self.num_heads)

        # obtain 'similarities' denote it as 'energy'
        # batch, num_heads, query
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min 
            # an object that represents the numerical properties of a floating point
            energy.mask_fill(~mask, fill_value)
        att = F.softmax(energy, dim= -1) * self.scaling 
        # similarities in prob by query
        att = self.att_drop(att)
        # dropout
        
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        # 'merge' the heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # b, n(num_patches+1), emb_size 
        out = self.projection(out)
        return out

# patches_embedded = PatchEmbedding()(x)
# out =MultiHeadAttention()(patches_embedded)
# print(out.shape)
    
# Transformer has residual connections
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion:int = 4, drop_p: float= 0.):
        super().__init__(
            nn.Linear(emb_size, expansion*emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, 
                 emb_size: int= 768,
                 drop_p: float= 0.,
                 forward_expansion: int= 4,
                 forward_drop_p: float = 0.,
                 **kwargs
                ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion= forward_expansion, 
                    drop_p= forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int= 12, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)]
        )
        self.depth = depth

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size:int= 768, n_classes:int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction= 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):

    def __init__(self, 
                 in_channels: int= 3,
                 patch_size: int= 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs
                ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size= emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.depth = depth
        self.n_classes = n_classes

summary(ViT(), (3, 224, 224), device= 'cpu')