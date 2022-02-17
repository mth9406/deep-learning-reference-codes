from re import M
from this import d
from unicodedata import bidirectional
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch
import torch.nn as nn

# input: embedded src_text
# output: ys: to calculate attention (bs, seq_len, hidden_size), h (2*num_layers, bs, hidden_size//2) 
class Encoder(nn.Module):
    
    def __init__(self, word_vec_dim, hidden_size, num_layers=4, dropout_p= .2):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0, 'Hidden size must be even...'
        
        half_hidden_size = hidden_size//2
        self.rnn = nn.LSTM(word_vec_dim, half_hidden_size, 
                           num_layers, 
                           dropout_p= dropout_p, 
                           bidirectional= True,
                           batch_first=True) # input_size, hidden_size, num_layers

    def forward(self, emb):
        # |emb| = (bs, seq_len, word_vec_dim)

        if isinstance(emb, tuple):
            x, lengths = emb
            # x: sentences
            # lengths: length of each sentence
            x = pack(x, lengths.tolist(), batch_first = True)
        else:
            x = emb
        
        y, h = self.rnn(emb)
        # feeds all the time stamp at the same time. 
        # |y| = (bs, seq_len, hidden_size) * bidirectional rnn.
        # h: (hidden_state, cell_state): tuple at the last time stamp.
        # |h[0]| = (num_layers * 2, bs, hidden_size//2)
        
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first= True)

        return y, h

class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim= -1)
                      
    def forward(self, h_src, h_t_tgt, mask= None):
        # |h_src| = (bs, seq_len, hidden_size)
        # |h_t_tgt| = (bs, 1, hidden_size) # feed one time stamp at a time.
        # |mask| = (bs, seq_len)
        
        query = self.linear(h_t_tgt)
        # |query| = (bs, 1, hidden_size)

        weight = torch.bmm(query, h_src.transpose(1,2))
        # |weight| = (bs, 1, seq_len)

        if mask is not None:
            # Set each weight as -inf if the mask value equals to 1.
            # Softmax(masked value) = 0
            # mask.unsqueeze(1) (bs, 1, seq_len)
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight, h_src)
        # |context_vector| = (bs, 1, hidden_size)

        return context_vector
        
class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers= 4, dropout_p = .2):
        super(Decoder, self).__init__()
        
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layer= n_layers,
            dropout_p= dropout_p,
            bidirectional= True,
            batch_first = True
        )
    
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (bs, 1, word_vec_size)
        # |h_t_l_tilde| = (bs, 1, hidden_size)
        # |h_t_l[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)
        
        if h_t_1_tilde is None:
            # the first time step
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()
            # use torsor.new() to create a torsor having the same type, shape, device

        # Input feeding trick
        x = torch.cat([emb_t, h_t_1_tilde], dim= -1) # (bs, 1, word_vec_size + hidden_size)
        
        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1)

        return y, h

class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (bs, seq_len, hidden_size)
        
        y = self.softmax(self.output(x))
        # Return log-probability 
        return y

class Seq2Seq(nn.Module):

    def __init__(self, 
                 input_size,
                 word_vec_size, 
                 hidden_size, 
                 output_size,
                 n_layers= 4,
                 dropout_p= 0.2    
                ):
        
        self.input_size = input_size # |V_src|
        self.word_vec_size = word_vec_size 
        self.hidden_size = hidden_size
        self.output_size = output_size # |V_tgt|
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super(Seq2Seq, self).__init__()  

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)
        
        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers = n_layers, dropout_p= dropout_p
        )

        self.decoder = Decoder(
            word_vec_size, hidden_size, 
            n_layers= n_layers, dropout_p = dropout_p
        )

        self.attn = Attention(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)
    
    def generate_mask(self, x, length):
        
        mask = []
        
        max_length = max(length) # max(sequence length) per batch
        # fill the mask
        # example
        # length = [4,3,1]
        # mask = [[0,0,0,0],[0,0,0,1],[0,1,1,1]]
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat(
                    [x.new_ones(1,l).zero_(),
                    x.new_ones(1, max_length-l)]
                ,dim=-1)]
            
            else:
                # l == max_length
                mask += [x.new_ones(1,l).zero_()]

        mask = torch.cat(mask, dim= 0).bool() 
        # (bs, max_length)
        # True if <PAD>
        return mask
    
    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens
        # |hiddens| = (n_layers*2, batch_size, hidden_size/2)
        # |cells| = (n_layers*2, batch_size, hidden_size/2)

        # parallelize the below for block
        # to make the code faster.
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i+1]], dim= -1)] # (forward+backward)
            new_cells += [torch.cat([cells[i], cells[i+1]], dim= -1)] # (forward+backward)

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return new_hiddens, new_cells

    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        # Merge bidirectional to uni-directional
        # we need to convert size from (n_layers*2, batch_size, hidden_size/2)
        # to (n_layers, batch_size, hidden_size)
        # Thus, the converting operation will not work with just "view" method.
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)
        # contiguous(): 
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()

        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()

        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        batch_size = src.size(0)

        # applies teacher forcing
        mask = None
        x_length = None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        
        if isinstance(tgt, tuple):
            tge = tgt[0]
        
        # Get word embedding vectors for every time step of input sentence.
        emb_src = self.emb_src(x)

        # The last hidden state of the encoder would be a initial hidden state of decoder
        h_src, h_0_tgt = self.encoder((emb_src), x_length)
        # |h_src| = (bs, seq_lengh, hidden_size)
        # |h_0_tgt[0]| = (num_layers, bs, hidden_size//2)

        h_0_tgt = self.fast_merge_encdoer_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (bs, seq_len, word_vec_size)
        h_tilde = []

        h_t_tilde = None
        # |h_t_tilde| = (bs, 1, hidden_size)
        decoder_hidden = h_0_tgt
        # Run decoder until the end of the time-stamp...
        for t in range(tgt.size(1)):
            # Teacher forcing
            # Do not use decoder_output as a new input to the next timestamp.
            emb_t = emb_tgt[:,t,:].unsqueeze(1) # (bs, 1, word_vec_size)
            
            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                        h_t_tilde,
                                                        decoder_hidden
                                                        )
            
            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                        context_vector], 
                                                        dim= -1)))
            # |h_t_tilde| = (bs, 1, hidden_size)
            h_tilde += [h_t_tilde]
        h_tilde = torch.cat(h_tilde, dim=1) # (bs, seq_len, hidden_size)
        y_hat = self.generator(h_tilde) # (bs, seq_len, output_size)

        return y_hat
            

