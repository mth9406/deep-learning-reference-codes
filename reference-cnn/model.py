from re import M
import torch
import torch.nn as nn

class Block(nn.Module):
    
    def __init__(self, 
                input_size, 
                output_size, 
                use_batch_norm = True,
                dropout_p = 0.4) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        return self.block(x)

class Conv2dBlock(nn.Module):
    
    def __init__(self, 
                in_channels, 
                out_channels,
                kernel_size,
                max_pool_channel=2) -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        super().__init__()
        
        # output size = floor( (input size - filter size + 2 * padding)/stride ) + 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(max_pool_channel)
        )

    def forward(self, x):
        return self.block(x)


class ImageClassifier(nn.Module):
    
    def __init__(self,
                output_size=10,
                dropout_p = 0.3) -> None:
        self.dropout_p = dropout_p
        self.output_size = output_size
        
        Blocks = [
            Conv2dBlock(1, 32, 3), # 32,14,14
            Conv2dBlock(32, 64, 3), # 64,7,7
            Block(7*7*64, 30)
        ]

        self.layers= nn.Sequential(
            *Blocks,
            nn.Linear(30, output_size),
            nn.LogSoftmax(dim=-1)
        )

        super().__init__()
        
            
    def forward(self, x):
        return self.layers(x)

                      