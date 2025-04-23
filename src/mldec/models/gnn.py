
"""notice: Much of this code is heavily ripped from https://github.com/LangeMoritz/GNN_decoder/tree/main"""
import torch
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import time

from torch_geometric.nn import global_mean_pool, GraphConv

import itertools
import gc
import stim
from mldec.utils.graph_representation import get_3D_graph


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
    Heavily ripped from Moritz Lange's GNN implementation.

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


# class GNN_Decoder:
#     """
#     Class for decoding syndromes with graph neural networks, with methods for
#     training the network continuously with graphs from random sampling of errors 
#     as training data. Samples are generated using multiprocessing.
#     """
#     def __init__(self, params = None):
#         # Set default parameters
#         self.params = {
#             'model': {
#                 'class': None,
#                 'num_node_features': 5,
#                 'num_classes': 1,
#                 'loss': None,
#                 'initial_learning_rate': 0.01,
#                 'manual_seed': 12345
#             },
#             'graph': {
#                 'm_nearest_nodes': None,
#                 'num_node_features': 5,
#                 'power': 2
#             },
#             'cuda': False,
#             'silent': False,
#             'save_path': './',
#             'save_prefix': None,
#             'resumed_training_file_name': None
#         }


#     ##########################################################
#     ###########  Method for data buffer training  ############
#     ##########################################################

#     def train_with_data_buffer(self, 
#             code_size,
#             repetitions,
#             error_rate, 
#             train = False,
#             save_to_file = False,
#             save_file_prefix = None,
#             num_iterations = 1,
#             batch_size = 200, 
#             buffer_size = 100,
#             replacements_per_iteration = 1,
#             test_size = 10000,
#             criterion = None, 
#             learning_rate = None,
#             benchmark = False,
#             learning_scheduler = False,
#             validation = False
#         ):
#         '''        
#         Train the decoder by generating a buffer of random syndrome graphs,
#         and continuously train the network with data from the buffer.
#         The true equivalence classes of the underlying errors are used
#         as training labels.
        
#         After each iteration, a number of batches in the buffer are replaced
#         by randomly sampling new graphs. Data is generated in parallel.

#         The input arguments 
#             replacements_per_iteration
#             buffer_size
#             batch_size
#             error_rate
#         determine how much data is taken from the buffer for training, and 
#         how much new data is put into the buffer with every iteration.
        
#         '''


#         # Get graph structure variables from parameter dictionary
#         num_node_features = params['graph']['num_node_features']
#         power = params['graph']['power']
#         cuda = params['cuda']
#         m_nearest_nodes = params['graph']['m_nearest_nodes']

#         criterion = torch.nn.BCEWithLogitsLoss()

#         sigmoid = torch.nn.Sigmoid() # To convert binary network output to class index
#         if cuda:
#             # Use GPU if available
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else: 
#             device = 'cpu'
        

#         # !!!!!!!!!!!!!!!!!!


#         ##############################################################################
#         ############################ TESTING (default)################################
#         ##############################################################################

#         if not train:
#             test_accuracy = generate_and_decode_test_batch(test_size)
#             return test_accuracy

#         ##############################################################################
#         ################################ TRAINING ####################################
#         ##############################################################################
#         if save_to_file:
#             print('Will save final results to file after training.')

#         print(f'Generating data then moving it to device {device}.')
#         print((f'Starting training with {num_iterations} iteration(s).'
#             f'\nBuffer has {buffer_size * batch_size * len(error_rate)} samples, replacing {replacements_per_iteration * batch_size * len(error_rate)} samples with each iteration.'
#             f'\nTotal number of unique samples in this run: {len(error_rate)*batch_size*(buffer_size+num_iterations*replacements_per_iteration):.2e}'))
#         previously_completed_samples = self.continuous_training_history['num_samples_trained']
#         if previously_completed_samples > 0:
#             print(f'Cumulative # of training samples from previous runs: {previously_completed_samples:.2e}')

#         # Store training parameters in history instance attribute
#         self.continuous_training_history['batch_size'] = batch_size
#         self.continuous_training_history['buffer_size'] = buffer_size
#         self.continuous_training_history['replacements_per_iteration'] = replacements_per_iteration
#         self.continuous_training_history['code_size'] = code_size
#         self.continuous_training_history['training_error_rate'] = error_rate
#         self.continuous_training_history['learning_rate'] = learning_rate


