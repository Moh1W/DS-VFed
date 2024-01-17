import math
import torch
from torch import nn
import torch.optim as optim

class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args
def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))

def split_data(dataset, worker_list=None, n_workers=2):
    '''
    split data vertically
    '''
    if worker_list is None:
        worker_list = list(range(0, n_workers))

    # counter to create the index of different data samples
    idx = 0

    # dictionary to accomodate the split data
    dic_single_datasets = {}
    for worker in worker_list:
        """
        Each value is a list of three elements, to accomodate, in order: 
        - data examples (as tensors)
        - label
        - index 
        """
        dic_single_datasets[worker] = []

    """
    Loop through the dataset to split the data and labels vertically across workers. 
    """
    label_list = []
    index_list = []
    for tensor, label in dataset:
        height = tensor.shape[-1] // len(worker_list)
        i = 0
        for worker in worker_list[:-1]:
            dic_single_datasets[worker].append(tensor[:, :, height * i: height * (i + 1)])
            i += 1

        # add the value of the last worker / split
        dic_single_datasets[worker_list[-1]].append(tensor[:, :, height * (i):])
        label_list.append(torch.Tensor([label]))
        index_list.append(torch.Tensor([idx]))

        idx += 1

    return dic_single_datasets, label_list, index_list


def clip(x, C):
    with torch.no_grad():
        L2_norm = torch.norm(x, p=2, dim=1)
        #         m=torch.mean(L2_norm,dim=0).item()

        for i in range(L2_norm.size(0)):
            if L2_norm[i].item() > C:
                x[i, :] = x[i, :] * (C * 1.0 / L2_norm[i].item())
    return x


def gaussion_noise(x, C, delta, epsilon):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    n = torch.normal(0, sigma * C, size=(x.size(0), x.size(1)))
    return n, sigma * C


def getrandomk_lap(score, x, k, a):
    '''
    score: laplacian score of each dimension
    '''
    mask = torch.ones_like(x)

    top_k_indices = torch.topk(score, k, largest=False).indices

    for i in range(x.shape[1]):
        if i not in top_k_indices:
            if torch.rand(1) > (1 - a):
                x[:, i] = 0
                mask[:, i] = 0
        else:
            if torch.rand(1) < (1 - a):
                x[:, i] = 0
                mask[:, i] = 0
    return x, mask


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError('Unknown optimizer type: {}.'.format(optimizer))
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'identity':
        return Identity()
    elif act_lower == 'relu':
        return nn.ReLU(inplace=False)
    elif act_lower == 'selu':
        return nn.SELU(inplace=False)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))
