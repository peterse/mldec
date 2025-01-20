import torch
import torch.nn as nn


class HiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        # self.act = nn.Tanh()
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.layer(x))


class FFNNlayered(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, input_dim, hidden_dim, output_dim, N_layers):
        super(FFNNlayered, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]*N_layers
        self.first =  HiddenLayer(input_dim, hidden_dim[0])
        self.layers = nn.ModuleList([HiddenLayer(hidden_dim[i], hidden_dim[i+1]) for i in range(N_layers-1)])
        self.output = nn.Linear(hidden_dim[-1], output_dim)
        self.act_output = nn.Sigmoid()
        # Custom initialization for the output layer
        self.init_output_layer()

    def init_output_layer(self):
        # Initialize weights to bias our output
        init_val = -0.1
        nn.init.constant_(self.output.weight, init_val)
        nn.init.constant_(self.output.bias, init_val)
        
    def forward(self, x):
        x = self.first(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return self.act_output(x)