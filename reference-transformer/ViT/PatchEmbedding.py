
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
# print(x.shape)

# braking down the image in multiple patches and flatten them...
patch_size = 16
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', 
                    s1= patch_size, s2= patch_size) # images to patches

# create a PatchEmbedding class to keep the code clean...
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int =3,
                 patch_size: int= 16,
                 emb_size: int= 768
                ):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.patch_size= patch_size
        self.emb_size= emb_size
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', 
                       s1=patch_size, s2= patch_size ),
            nn.Linear(patch_size*patch_size*in_channels,
                        emb_size)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)

print(PatchEmbedding()(x).shape)

# PatchEmbedding using a convolution layer
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int =3,
                 patch_size: int= 16,
                 emb_size: int= 768
                ):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.patch_size= patch_size
        self.emb_size= emb_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, patch_size),
            # encode patches 
            Rearrange('b e h w -> b (h w) e')
            # flatten the patches 
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)

print(PatchEmbedding()(x).shape)

# CLS Token 

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int= 3,
                 patch_size: int= 16,
                 emb_size: int= 768
                 ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))

    def forward(self, x:Tensor) -> Tensor:
        x = self.projection(x)
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # pad cls_tokens to every obs
        x = torch.cat([cls_tokens, x], dim= 1) 
        return x

print(PatchEmbedding()(x).shape)        

# Position Embedding
# Position embedding is just a tensor of shape
# of (patches+1 token, embed_size) that is added to
# the projected patches...

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int= 3,
                 patch_size: int= 16, emb_size: int= 768,
                 img_size: int = 224):
        self.patch_size= patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 
                        kernel_size= patch_size, stride= patch_size),
            Rearrange('b e h w -> b (h w) e') # flatten the patches; batch, number of pathes, emb_size 
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # concat the cls_token -> # batch, 1 + number of pathes, emb_size
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 +1, emb_size))
        # number of patches, emb_size

    def forward(self, x:Tensor) -> Tensor:
        x = self.projection(x)
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # batch, 1, emb_size
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim= 1)
        x += self.positions
        return x

print(PatchEmbedding()(x).shape)

