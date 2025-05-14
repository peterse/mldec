
"""GNN code."""
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool, GraphConv


class RepGNN(nn.Module):
    """Wrapper class for a simple fully-connected feed-forward neural network
    
    exampls parameters
    gcn_depth = 5
    gcn_min = 32
    mlp_depth = 3
    mlp_max = 32
    
    Results in (32, 64, 128, 64, 32) for GCN layers and (32, 16, 8) for MLP layers.

    Args:
        input_dim: int, the dimension of the input = syndrome length
        output_dim: int, the dimension of the output = output length.
        gcn_depth: number of GCN layers
        gcn_min: minimum channel count for the first layer, which is doubled for gcn_depth/2 layers then halved again
        mlp_depth: number of MLP layers
        mlp_max: maximum channel count for the first layer, which is halved for mlp_depth layers
    """
    def __init__(self, input_dim, output_dim, gcn_depth, gcn_min, mlp_depth, mlp_max, device=None):
        super(RepGNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        gcn_layers = [gcn_min]
        max_pt = (gcn_depth)// 2
        for i in range(max_pt):
            gcn_layers.append(gcn_layers[-1] * 2)
        for i in range(max_pt, gcn_depth - 1):
            gcn_layers.append(gcn_layers[-1] // 2)
            
        mlp_layers = [mlp_max]
        for i in range((mlp_depth - 1)):
            mlp_layers.append(mlp_layers[-1] // 2)
        self.gcn_layers = gcn_layers
        self.mlp_layers = mlp_layers
        self.device = device

        self._initialize_model()        

    def _initialize_model(self):
        self.model = GNN_flexible(
            hidden_channels_GCN=self.gcn_layers,
            hidden_channels_MLP=self.mlp_layers,
            num_node_features=self.input_dim, 
            num_classes=self.output_dim).to(self.device)
        
    def training_step(self, data, optimizer, criterion):
        """Perform a single training step; `data` represents a batch of data
        data.x            (number of nodes in sample * batch_size, 4)
        data.edge_index   (2, number of edge_indices in sample * batch_size)
        data.edge_attr    (number of edge_indices in sample * batch_size, 1)
        data.y            (batch_size, 2 for two-head 4 for one-head)
        """
        
        optimizer.zero_grad()
        data.batch = data.batch.to(self.device)
        out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        target = data.y.to(int)
        # print(f"target: {target.reshape(-1)}")
        # print(f"out: {out.reshape(-1)}")
        loss = criterion(out, data.y)
        prediction = (torch.sigmoid(out.detach()) > 0.5).to(self.device).long()
        correct_predictions = int((prediction == target).sum().item())

        loss.backward()
        optimizer.step()
        return correct_predictions, loss

    def predict(self, data):
        """Predict the output of the model."""
        data.batch = data.batch.to(self.device)
        out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        prediction = torch.sigmoid(out.detach()).round().to(int)
        return prediction


class GNN_flexible(torch.nn.Module):
    '''
    Heavily borrowed from Moritz Lange's GNN implementation https://arxiv.org/pdf/2307.01241.
    
    Copyright (c) 2023 Moritz Lange (MIT License)

    GNN with a flexible number of GraphConv layers, whose final output is converted
    to a single graph embedding (feature vector) with global_mean_pool.
    This graph embedding is duplicated, and passed to a
    dense network which performs binary classification.
    The binary classifications represents the logical equivalence classes 
    X or Z, respectively.
    The output of this network is a tensor with 1 element, giving a binary
    representation of the predicted equivalence class.
    0 <-> class I
    1 <-> class X or Z
    '''
    def __init__(self,
                #  hidden_channels_GCN=(32, 128, 256, 512, 512, 256, 256),
                #  hidden_channels_MLP=(256, 128, 64),
                 hidden_channels_GCN=(32, 64, 128, 64, 32),
                 hidden_channels_MLP=(32, 16),
                 num_node_features=5, 
                 num_classes=1, 
                #  manual_seed=1234,
                 ):
        super().__init__()

        # Build GraphConv layers using ModuleList
        self.convs = nn.ModuleList()
        in_channels = num_node_features
        for out_channels in hidden_channels_GCN:
            self.convs.append(GraphConv(in_channels, out_channels))
            in_channels = out_channels

        # Build MLP layers
        mlp_layers = []
        prev_dim = hidden_channels_GCN[-1]
        for hidden_dim in hidden_channels_MLP:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.ModuleList(mlp_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        # Pass through each GCN layer in a loop
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        # Pass through the MLP
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:  # apply ReLU except for final output
                x = F.relu(x)
        return x