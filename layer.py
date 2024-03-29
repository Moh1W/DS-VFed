import torch.nn as nn
import torch
import math
from utils import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]

class FeatureSelector(nn.Module):
    def __init__(self, sigma, mu):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(mu, requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)
        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu',is_top=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []
        if(is_top==False):
            for i in range(len(dims)-1):
                layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
                modules.append(layer)
            self.mlp = nn.Sequential(*modules)
        else:
            for i in range(len(dims)-2):
                layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
                modules.append(layer)
            layer = nn.Linear(dims[-2], dims[-1], bias=True)
            modules.append(layer)
            self.mlp = nn.Sequential(*modules)


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        # if self.flatten:
        #     input = input.view(input.size(0), -1)
        return self.mlp(input)
