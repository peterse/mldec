import torch.nn as nn

class HiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(HiddenLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.act(self.layer(x))
        return self.dropout(x)


class FFNN(nn.Module):
    """Wrapper class for a simple fully-connected feed-forward neural network
    
    Args:
        input_dim: int, the dimension of the input = syndrome length
        hidden_dim: int or list of ints, the dimension of the hidden layers
        output_dim: int, the dimension of the output = output length.
        n_layers: int, the number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, N_layers, dropout=0, device=None):
        super(FFNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.N_layers = N_layers
        self.dropout = dropout
        self.device = device

        self._initialize_model()        

    def _initialize_model(self):
        self.model = FFNNmodel(
            self.input_dim, 
            self.hidden_dim, 
            self.output_dim, 
            self.N_layers, 
            self.dropout).to(self.device)
        
    def training_step(self, X, Y, weights, optimizer, criterion):
        """Perform a single training step."""
        optimizer.zero_grad()
        Y_pred = self.model(X)
        loss = criterion(Y_pred, Y, weights)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, X):
        """Predict the output of the model."""
        activations = self.model(X)
        return (activations >= 0).float()


class FFNNmodel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, N_layers, dropout=0):
        super(FFNNmodel, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]*N_layers
        self.first =  HiddenLayer(input_dim, hidden_dim[0], dropout)
        self.layers = nn.ModuleList([HiddenLayer(hidden_dim[i], hidden_dim[i+1], dropout) for i in range(N_layers-1)])
        self.output = nn.Linear(hidden_dim[-1], output_dim)
        # self.act_output = nn.Sigmoid() # assume BCEWithLogitsLoss
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
        return self.output(x)
        return self.act_output(x)
