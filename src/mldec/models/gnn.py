
"""notice: Much of this code is heavily ripped from https://github.com/LangeMoritz/GNN_decoder/tree/main"""
import torch
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, GraphConv

import itertools
import gc
import stim
from mldec.utils.graph_representation import get_3D_graph




class RepGNN(nn.Module):
    """Wrapper class for a simple fully-connected feed-forward neural network
    
    Args:
        input_dim: int, the dimension of the input = syndrome length
        hidden_dim: int or list of ints, the dimension of the hidden layers
        output_dim: int, the dimension of the output = output length.
        n_layers: int, the number of hidden layers
    """
    def __init__(self, gcn_layers, mlp_layers, dropout=0, device=None):
        super(RepGNN, self).__init__()
        
        self.gcn_layers = gcn_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.device = device

        self._initialize_model()        

    def _initialize_model(self):
        self.model = GNN_flexible(
            hidden_channels_GCN=self.gcn_layers,
            hidden_channels_MLP=self.mlp_layers,
            dropout=self.dropout).to(self.device)
        
    def training_step(self, X, Y, weights, optimizer, criterion):
        """Perform a single training step."""
        raise NotImplementedError
        optimizer.zero_grad()
        Y_pred = self.model(X)
        loss = criterion(Y_pred, Y, weights)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, X):
        raise NotImplementedError
        """Predict the output of the model."""
        activations = self.model(X)
        return (activations >= 0).float()


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
                 manual_seed=1234):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)

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


class GNN_Decoder:
    """
    Class for decoding syndromes with graph neural networks, with methods for
    training the network continuously with graphs from random sampling of errors 
    as training data. Samples are generated using multiprocessing.
    """
    def __init__(self, params = None):
        # Set default parameters
        self.params = {
            'model': {
                'class': None,
                'num_node_features': 5,
                'num_classes': 1,
                'loss': None,
                'initial_learning_rate': 0.01,
                'manual_seed': 12345
            },
            'graph': {
                'm_nearest_nodes': None,
                'num_node_features': 5,
                'power': 2
            },
            'cuda': False,
            'silent': False,
            'save_path': './',
            'save_prefix': None,
            'resumed_training_file_name': None
        }


    ##########################################################
    ###########  Method for data buffer training  ############
    ##########################################################

    def train_with_data_buffer(self, 
            code_size,
            repetitions,
            error_rate, 
            train = False,
            save_to_file = False,
            save_file_prefix = None,
            num_iterations = 1,
            batch_size = 200, 
            buffer_size = 100,
            replacements_per_iteration = 1,
            test_size = 10000,
            criterion = None, 
            learning_rate = None,
            benchmark = False,
            learning_scheduler = False,
            validation = False
        ):
        '''        
        Train the decoder by generating a buffer of random syndrome graphs,
        and continuously train the network with data from the buffer.
        The true equivalence classes of the underlying errors are used
        as training labels.
        
        After each iteration, a number of batches in the buffer are replaced
        by randomly sampling new graphs. Data is generated in parallel.

        The input arguments 
            replacements_per_iteration
            buffer_size
            batch_size
            error_rate
        determine how much data is taken from the buffer for training, and 
        how much new data is put into the buffer with every iteration.
        
        '''

        ##########################################################
        ######################### Setup  #########################
        ##########################################################

        if benchmark:
            time_sample = 0.
            time_fit = 0.
            time_setup_start = time.perf_counter()

        if save_to_file:
            if save_file_prefix is None:
                if self.params['save_prefix'] is None:
                    print(('No filename was given.'
                    '\nSpecify a filename with the prefix and suffix arguments. '
                    '\nAlternatively, specify the save_prefix class instance '
                    'attribute before calling this method.'))
                    return

        params = self.params
        model = self.model
        optimizer = self.optimizer

        # If learning rate is not specified, use current learning rate
        # of optimizer (0.01 if initialized by default).
        if learning_rate is None:
            learning_rate = optimizer.param_groups[0]['lr'] 
        optimizer.param_groups[0]['lr'] = learning_rate

        # Get graph structure variables from parameter dictionary
        num_node_features = params['graph']['num_node_features']
        power = params['graph']['power']
        cuda = params['cuda']
        m_nearest_nodes = params['graph']['m_nearest_nodes']

        criterion = torch.nn.BCEWithLogitsLoss()

        sigmoid = torch.nn.Sigmoid() # To convert binary network output to class index
        if cuda:
            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: 
            device = 'cpu'
        
        # Initialize stim circuit for a fixed training rate
        circuit = stim.Circuit.generated(
                    "surface_code:rotated_memory_z",
                    rounds = repetitions,
                    distance = code_size,
                    after_clifford_depolarization = error_rate,
                    after_reset_flip_probability = error_rate,
                    before_measure_flip_probability = error_rate,
                    before_round_data_depolarization = error_rate)
        # get detector coordinates (same for all error rates):
        detector_coordinates = circuit.get_detector_coordinates()

        # get coordinates of detectors (divide by 2 because stim labels 2d grid points)
        # coordinates are of type (d_west, d_north, hence the reversed order)
        detector_coordinates = np.array(list(detector_coordinates.values()))

        # rescale space like coordinates:
        detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2

        # convert to integers
        detector_coordinates = detector_coordinates.astype(np.uint8)
        sampler = circuit.compile_detector_sampler()




        ####################################################################
        # DEFINE HELPER FUNCTIONS
        ####################################################################
        def syndrome_mask(code_size, repetitions):
            '''
            Creates a surface code grid. 1: X-stabilizer. 3: Z-stabilizer.
            '''
            M = code_size + 1

            syndrome_matrix_X = np.zeros((M, M), dtype = np.uint8)

            # starting from northern boundary:
            syndrome_matrix_X[::2, 1:M - 1:2] = 1

            # starting from first row inside the grid:
            syndrome_matrix_X[1::2, 2::2] = 1

            syndrome_matrix_Z = np.rot90(syndrome_matrix_X) * 3
            # Combine syndrome matrices where 1 entries 
            # correspond to x and 3 entries to z defects
            syndrome_matrix = (syndrome_matrix_X + syndrome_matrix_Z)

            # Return the syndrome matrix
            return np.dstack([syndrome_matrix] * (repetitions + 1))
        
        # get the surface code grid:
        mask = syndrome_mask(code_size, repetitions)
    
        def generate_buffer(buffer_size):
            '''
            Creates a buffer with len(error_rate)*batch_size*buffer_size samples.
            Empty syndromes are removed from the training data, because empty graphs 
            can't be handled by PyTorch and should be easily classified as I.
            '''
            stim_data_list, observable_flips_list = [], []

            # this is a resampling factor that I am reusing. it doesn't hurt to be larger i guess
            factor = max(1/(20*error_rate), 10)
            # sample for each error rate:
            stim_data_one_p, observable_flips_one_p = [], []
            while len(stim_data_one_p) < (batch_size*buffer_size):
                stim_data, observable_flips = sampler.sample(shots = factor*batch_size*buffer_size, separate_observables=True)
                # remove empty syndromes:
                non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
                stim_data_one_p.extend(stim_data[non_empty_indices, :])
                observable_flips_one_p.extend(observable_flips[non_empty_indices])
            # if there are more non-empty syndromes than necessary
            stim_data_list.append(stim_data_one_p[: batch_size*buffer_size])
            observable_flips_list.append(observable_flips_one_p[: batch_size*buffer_size])

            # interleave lists to mix error rates: 
            # [sample(p1), sample(p2), ..., sample(p_n), sample(p1), sample(p2), ...]
            # FIXME: cut these out, but retain the correct shape
            stim_data_list = [val for tup in zip(*stim_data_list) for val in tup]
            observable_flips_list = [val for tup in zip(*observable_flips_list) for val in tup]

            # len of single batches:
            # N_b = no_samples / buffer_size = len(error_rate) * batch_size
            repeated_arguments = []
            N_b = batch_size * len(error_rate)
            for i in range(buffer_size):
                repeated_arguments.append((stim_data_list[i * N_b : (i + 1) * N_b],
                                          observable_flips_list[i * N_b : (i + 1) * N_b],
                                          detector_coordinates,
                                          mask, m_nearest_nodes, power))
            # create batches in parallel:
            buffer = itertools.starmap(generate_batch, repeated_arguments)

            # flatten the buffer:
            buffer = [item for sublist in buffer for item in sublist]
    
            # convert list of numpy arrays to torch Data object containing torch GPU tensors
            torch_buffer = []
            for i in range(len(buffer)):
                X = torch.from_numpy(buffer[i][0]).to(device)
                edge_index = torch.from_numpy(buffer[i][1]).to(device)
                edge_attr = torch.from_numpy(buffer[i][2]).to(device)
                y = torch.from_numpy(buffer[i][3]).to(device)
                torch_buffer.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y = y))

            return torch_buffer

        def update_buffer(buffer, replacements_per_iteration):
            '''
            Replaces the oldest samples from the buffer
            with new samples.
            '''
            # delete the first entries of the buffer:
            del buffer[: (replacements_per_iteration * batch_size * len(error_rate))]
            # create new data:
            new_data = generate_buffer(replacements_per_iteration)
            buffer.extend(new_data)
            return buffer

        def train_with_buffer(graph_list, shuffle=True):
            '''Trains the network with data from the buffer.'''
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)
            total_loss = 0.
            correct_predictions = 0
            model.train()
            # tensor sizes: 
            # data.x            (number of nodes in sample * batch_size, 4)
            # data.edge_index   (2, number of edge_indices in sample * batch_size)
            # data.edge_attr    (number of edge_indices in sample * batch_size, 1)
            # data.y            (batch_size, 2 for two-head 4 for one-head)
            for data in loader:
                optimizer.zero_grad()
                data.batch = data.batch.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                # don't include the basis label as target:
                target = data.y.to(int)

                # print(out.shape, data.y.shape)
                loss = criterion(out, data.y)
                prediction = (sigmoid(out.detach()) > 0.5).to(device).long()
                correct_predictions += int((prediction == target).sum().item())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            return correct_predictions, total_loss
        
        def generate_test_batch(test_size):
            '''Generates a test batch at one test error rate'''
            # Keep track of trivial syndromes
            correct_predictions_trivial = 0
            stim_data_list, observable_flips_list = [], []

            # repeat each experiments multiple times to get enough non-empty 
            # syndromes. This number decreases with increasing p
            stim_data, observable_flips = sampler.sample(shots =  test_size, separate_observables = True)
            # remove empty syndromes:
            # (don't count imperfect X(Z) in second to last time)
            non_empty_indices = (np.sum(stim_data, axis = 1) != 0)
            stim_data_list.extend(stim_data[non_empty_indices, :])
            observable_flips_list.extend(observable_flips[non_empty_indices])
            # count empty instances as trivial predictions: 
            correct_predictions_trivial += len(observable_flips[~ non_empty_indices])
            # if there are more non-empty syndromes than necessary
            stim_data_list = stim_data_list[: test_size]
            observable_flips_list = observable_flips_list[: test_size]
            buffer = generate_batch(stim_data_list, observable_flips_list,
                                    detector_coordinates, mask, m_nearest_nodes, power)
            # convert list of numpy arrays to torch Data object containing torch GPU tensors
            test_batch = []
            for i in range(len(buffer)):
                X = torch.from_numpy(buffer[i][0]).to(device)
                edge_index = torch.from_numpy(buffer[i][1]).to(device)
                edge_attr = torch.from_numpy(buffer[i][2]).to(device)
                y = torch.from_numpy(buffer[i][3]).to(device)
                test_batch.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y = y))
            return test_batch, correct_predictions_trivial

        def count_correct_predictions_in_test_batch(graph_batch):
            '''Counts the correct predictions by the network for a test batch'''
            loader = DataLoader(graph_batch, batch_size = 1000)
            correct_predictions = 0
            model.eval()            # run network in training mode 
            with torch.no_grad():   # turn off gradient computation (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
                for data in loader:
                    # Perform forward pass to get network output
                    data.batch = data.batch.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    target = data.y.to(int) # Assumes binary targets (no probabilities)
                    # Sum correct predictions
                    prediction = sigmoid(out.detach()).round().to(int)
                    correct_predictions += int( (prediction == target).sum() )

            return correct_predictions
        
        def decode_test_batch(graph_batch):
            '''
            Estimates the decoder's logical success rate with one batch 
            of test graphs which were generated in advance. Returns the accuracy
            tested without trivial syndromes ('val_accuracy').
            '''
            # Count correct predictions by GNN for nontrivial syndromes
            correct_predictions_nontrivial = count_correct_predictions_in_test_batch(graph_batch)
            val_accuracy = correct_predictions_nontrivial / len(graph_batch)

            return val_accuracy

        def generate_and_decode_test_batch(test_size):
            '''
            Estimates the decoder's logical success rate with one batch 
            of test graphs which are generated with this function. Returns the 
            accuracy tested with trivial syndromes.
            '''
            # Generate a test batch
            graph_batch, correct_predictions_trivial = generate_test_batch(test_size)
            # Count correct predictions by GNN for nontrivial syndromes
            correct_predictions_nontrivial = count_correct_predictions_in_test_batch(graph_batch)
            test_accuracy = (correct_predictions_nontrivial + correct_predictions_trivial) / test_size
            print(f'non-trivial syndromes: {len(graph_batch)}, trivial syndromes: {correct_predictions_trivial}')
            print(f'correct, non-trivial predictions: {correct_predictions_nontrivial}')
            return test_accuracy

        ##############################################################################
        ############################ TESTING (default)################################
        ##############################################################################

        if not train:
            test_accuracy = generate_and_decode_test_batch(test_size)
            return test_accuracy

        ##############################################################################
        ################################ TRAINING ####################################
        ##############################################################################
        if save_to_file:
            print('Will save final results to file after training.')

        print(f'Generating data then moving it to device {device}.')
        print((f'Starting training with {num_iterations} iteration(s).'
            f'\nBuffer has {buffer_size * batch_size * len(error_rate)} samples, replacing {replacements_per_iteration * batch_size * len(error_rate)} samples with each iteration.'
            f'\nTotal number of unique samples in this run: {len(error_rate)*batch_size*(buffer_size+num_iterations*replacements_per_iteration):.2e}'))
        previously_completed_samples = self.continuous_training_history['num_samples_trained']
        if previously_completed_samples > 0:
            print(f'Cumulative # of training samples from previous runs: {previously_completed_samples:.2e}')

        # Store training parameters in history instance attribute
        self.continuous_training_history['batch_size'] = batch_size
        self.continuous_training_history['buffer_size'] = buffer_size
        self.continuous_training_history['replacements_per_iteration'] = replacements_per_iteration
        self.continuous_training_history['code_size'] = code_size
        self.continuous_training_history['training_error_rate'] = error_rate
        self.continuous_training_history['learning_rate'] = learning_rate


        # time for training setup:
        if benchmark:
            time_setup_end = time.perf_counter()
            sample_start = time.perf_counter()

        # Initialize list of validation accuracies if it does not yet exist and generate validation batch
        if validation:
            # Initialize data buffer
            data_buffer = generate_buffer(buffer_size + test_size)
            test_val_batch = data_buffer[:(test_size * batch_size * len(error_rate))]
            data_buffer = data_buffer[(test_size * batch_size * len(error_rate)):]
            gc.collect()
            try:
                self.continuous_training_history['val_acc'] == []
            except KeyError:
                self.continuous_training_history['val_acc'] = []
        else:
            data_buffer = generate_buffer(buffer_size)
            gc.collect()
        # time for sampling:
        if benchmark:
            sample_end = time.perf_counter()
            time_sample += (sample_end - sample_start)
        
        
        # If >100 iterations, limit number of progress prints to 100
        if num_iterations > 100: 
            training_print_spacing = int(num_iterations/100)


        for ix in range(num_iterations):
            if benchmark:
                fit_start = time.perf_counter()
            # forward pass:
            correct_count, total_iteration_loss = train_with_buffer(data_buffer)
            sample_count = len(data_buffer)
            # Benchmarking
            if benchmark:
                fit_end = time.perf_counter()
                time_fit += (fit_end - fit_start)
                sample_start = time.perf_counter()
            # update the data buffer (either replace or append batches)
            if replacements_per_iteration > 0:
                data_buffer = update_buffer(data_buffer, replacements_per_iteration)
            gc.collect()
            
            # Store loss and accuracy from training iteration
            average_sample_loss = total_iteration_loss / sample_count
            iteration_accuracy = correct_count / sample_count

            # Benchmarking
            if benchmark:
                sample_end = time.perf_counter()
                time_sample += (sample_end - sample_start)

            # If validation, test on validation batch
            if validation:
                val_acc = decode_test_batch(test_val_batch)
                self.continuous_training_history['val_acc'].append(val_acc)
            # save training iteration metrics
            self.continuous_training_history['loss'].append(average_sample_loss)
            self.continuous_training_history['accuracy'].append(iteration_accuracy)
            self.continuous_training_history['num_samples_trained'] += sample_count
            # Print training results. If running >100 iterations, increase spacing of prints
            if (num_iterations <= 100) or ((ix+1) % training_print_spacing == 0) or (ix == 0):
                training_print_string = (f'Iteration: {(ix+1):03d}\t'
                                        f'Loss: {average_sample_loss:.4f}\t'
                                        f'Acc: {iteration_accuracy:.4f}\t')
                if validation:
                    training_print_string += f'Validation Acc: {val_acc:.4f}\t'
                training_print_string += (f'Cumulative # of training samples: '
                                        f'{(previously_completed_samples + sample_count * (ix+1)):.2e}')
                print(training_print_string)

        print('Completed all training iterations!')
        if save_to_file:
            print('Saving final model and history to file.')
            self.save_attributes_to_file(
                prefix = save_file_prefix, 
                suffix = '')

        # BENCHMARKING
        if benchmark:
            print('\n==== BENCHMARKING ====')
            time_setup = time_setup_end - time_setup_start
            print(f'Training setup: {time_setup}')
            print(f'Sampling and Graphing: {time_sample}')
            print(f'Fitting: {time_fit}')
            print(f'\tSum: {time_setup + time_sample + time_fit}')
        
        return

def generate_batch(stim_data_list,
                   observable_flips_list,
                   detector_coordinates,
                   mask, m_nearest_nodes, power):
    '''
    Generates a batch of graphs from a list of stim experiments.
    '''
    batch = []

    for i in range(len(stim_data_list)):
        # convert to syndrome grid:
        syndrome = stim_to_syndrome_3D(mask, detector_coordinates, stim_data_list[i])
        # get the logical equivalence class:
        true_eq_class = np.array([int(observable_flips_list[i])])
        # map to graph representation
        graph = get_3D_graph(syndrome_3D = syndrome,
                             target = true_eq_class,
                             power = power,
                             m_nearest_nodes = m_nearest_nodes)
        batch.append(graph)
    return batch

def stim_to_syndrome_3D(mask, coordinates, stim_data):
    '''
    Converts a stim detection event array to a syndrome grid. 
    1 indicates a violated X-stabilizer, 3 a violated Z stabilizer. 
    Only the difference between two subsequent cycles is stored.
    '''
    # initialize grid:
    syndrome_3D = np.zeros_like(mask)

    # first to last time-step:
    syndrome_3D[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = stim_data

    # only store the difference in two subsequent syndromes:
    syndrome_3D[:, :, 1:] = (syndrome_3D[:, :, 1:] - syndrome_3D[:, :, 0: - 1]) % 2

    # convert X (Z) stabilizers to 1(3) entries in the matrix
    syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]

    return syndrome_3D