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
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        return self.block(x)

class ImageClassifier(nn.Module):
    
    def __init__(self,
                input_size, 
                output_size, 
                hidden_sizes= [500, 400, 300, 200, 100],
                use_batch_norm = True,
                dropout_p = 0.3) -> None:

        assert len(hidden_sizes) > 0, 'Hidden layers should be specified.'
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.layers = nn.Sequential(
            *list(self.generate_layers(hidden_sizes)),
            nn.Linear(hidden_sizes[-1], output_size),
            nn.LogSoftmax(dim= -1)
        )
        
    def generate_layers(self, hidden_sizes):

        for i in range(len(hidden_sizes)):
            if i != 0:
                yield Block(hidden_sizes[i-1], hidden_sizes[i], 
                            self.use_batch_norm, dropout_p= self.dropout_p)
            else:
                yield Block(self.input_size, hidden_sizes[i], 
                            self.use_batch_norm, dropout_p= self.dropout_p)
    
    def forward(self, x):
        return self.layers(x)

                      