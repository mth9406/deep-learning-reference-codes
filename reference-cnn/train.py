import argparse
from doctest import OutputChecker

import torch  
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
# from utils import get_hidden_sizes

# parse the arguments
def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required= True)
    p.add_argument('--gpu_id', type = int, default= 0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type= float, default= .8)

    p.add_argument('--batch_size', type= int, default= 256)
    p.add_argument('--n_epochs', type= int, default= 10)
    # p.add_argument('--n_layers', type= int, default = 5)

    # p.add_argument('--use_dropout', action='store_true') # depreciated
    # p.add_argument('--dropout_p', type= float, default= 0.3)  # depreciated

    p.add_argument('--verbose', type= int, default= 1)

    # For the future use
    p.add_argument('--is_train', type = bool, default = True)
    p.add_argument('--flatten', type = bool, default= False)

    config = p.parse_args()

    return config

def main(config):
    # set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train= True, flatten= False)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    # input_size = int(x[0].shape[-1])
    # output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        output_size= 10,
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)
    
    trainer = Trainer(model, optimizer, crit)
    
    trainer.train(
        train_data= (x[0], y[0]),
        valid_data = (x[1], y[1]),
        config = config
    )

    # Save the best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)