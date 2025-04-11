import torch
import torch.nn as nn

class CNN(nn.Module):
    """Wrapper for a simple 1D CNN
    
    Args:
        input_dim: int, the length T of the 1D input
        conv_channels: int or list of ints, output channels for each conv layer
        output_dim: int, the final output dimension [B, output_dim]
        n_layers: int, number of convolutional layers
        dropout: float, dropout probability
        device: torch.device or str
    """
    def __init__(self, input_dim, conv_channels, output_dim,
                 n_layers, kernel_size, dropout=0, device=None):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device

        self._initialize_model()

    def _initialize_model(self):
        self.model = CNNmodel(
            input_dim=self.input_dim,
            conv_channels=self.conv_channels,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

    def training_step(self, X, Y, weights, optimizer, criterion):
        """Perform a single training step with VIRTUAL DATA."""
        optimizer.zero_grad()
        Y_pred = self.model(X)
        loss = criterion(Y_pred, Y, weights)
        loss.backward()
        optimizer.step()
        return loss

    def real_training_step(self, X, Y, optimizer, criterion):
        """Perform a single training step with REAL data."""
        optimizer.zero_grad()
        Y_pred = self.model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, X):
        """Predict the output of the model."""
        activations = self.model(X)
        return (activations >= 0).float()
        return self.model(X)


class CNNmodel(nn.Module):
    def __init__(self, input_dim, conv_channels, output_dim, n_layers, kernel_size, dropout=0):
        super(CNNmodel, self).__init__()
        
        # If conv_channels is a single int, replicate it for n_layers
        if isinstance(conv_channels, int):
            conv_channels = [conv_channels] * n_layers

        layers = []
        in_ch = 1  # Because we reshape x to [B, 1, T]
        for out_ch in conv_channels:
            # Convolution with padding to preserve length
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=1))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers) #[B, conv_channels[-1], T]

        # We'll apply a global average pool over T => [B, conv_channels[-1]].
        self.output_layer = nn.Linear(conv_channels[-1], output_dim)
        self.init_output_layer()

    def init_output_layer(self):
        # Example initialization to bias the output
        init_val = -0.1
        nn.init.constant_(self.output_layer.weight, init_val)
        nn.init.constant_(self.output_layer.bias, init_val)

    def forward(self, x):
        # x is [B, T]. Reshape to [B, 1, T]
        x = x.unsqueeze(1)
        x = self.conv_layers(x)            # => [B, conv_channels[-1], T]
        x = x.mean(dim=-1)                 # global average pool => [B, conv_channels[-1]]
        x = self.output_layer(x)           # => [B, output_dim]
        return x
