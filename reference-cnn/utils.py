import torch

# a function to load data.
# is_train: to download training samples
# flatten: depreciated when CNN layers are tobe used.
def load_mnist(is_train= True, flatten= True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        './data', train= is_train, download = True,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    x = dataset.data.float() / 255. # normalize the data
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1) # 28 * 28
    else:
        x = x[:, None, :, :] # numObs, 1, 28, 28

    return x, y

# shuffles the data
# and returns x[training, valid], y[training, valid]
def split_data(x, y, train_ratio= 0.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = int(x.size(0)) - train_cnt

    # shuffles dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y

# def get_hidden_sizes(input_size, output_size, n_layers):
#     # creates hidden layers in an arithmetical way.
#     step_size = int((input_size-output_size)/ n_layers)
#     # example:
#     # input_size = 100, n_layers = 3, ouput_size= 10
#     # (100-10)//3 = 90//3 = 30
#     # 100 --> 70 --> 40 --> 10!
#     # hiddens = [70, 40]

#     hidden_sizes = []
#     current_size = input_size - step_size
#     while current_size > output_size:
#         hidden_sizes.append(current_size)
#         current_size -= step_size
    
#     return hidden_sizes


