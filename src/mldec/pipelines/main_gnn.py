import numpy as np
import torch
import os
from torch_geometric.loader import DataLoader
from mldec.models.gnn import GNN_flexible, GNN_Decoder

if __name__ == '__main__':
    # Code and noise settings
    code_size = 3
    training_error_rate = [0.005]
    repetitions = 3

    # Training settings
    num_iterations = 10
    batch_size = 32
    buffer_size = 4

    # total training data is batch_size * buffer_size * len(training_error_rate)
    learning_rate = 0.0003
    criterion = torch.nn.BCEWithLogitsLoss()
    manual_seed = 1234
    benchmark = False
    replacements_per_iteration = 1
    test_size = 100

    # Graph settings
    # num_node_features = 5
    power = 2
    m_nearest_nodes = None


    cuda = False
    validation = True

    # IO settings
    job_name = "gnn_test"
    job_id = 0
    save_directory_path = os.path.join(".", 'train_results/') # Add node local results dir to path
    filename_prefix = f'{job_name}_id{job_id}' 
    # If specified, resume run by loading Decoder attributes from file (history and model/optim state dicts)
    resumed_training_file_name = None

    GNN_params = {
        'model': {
            'class': GNN_flexible,
            'num_classes': 1, # 1 output class for two-headed model
            'loss': criterion,
            'num_node_features': num_node_features,
            'initial_learning_rate': learning_rate,
            'manual_seed': manual_seed
        },
        'graph': {
            'num_node_features': num_node_features,
            'm_nearest_nodes': m_nearest_nodes,
            'power': power
        },
        'cuda': cuda,
        'save_path': save_directory_path, 
        'save_prefix': filename_prefix
    }

    # INITIALIZE DECODER, SET PARAMETERS
    print('\n==== DECODER PARAMETERS ====')
    decoder = GNN_Decoder(GNN_params)
    print(decoder.params)
    print(f'Code size: {code_size}\n')
    print(f'Repetitions: {repetitions}\n')
    print(f'Training error rate: {training_error_rate}\n')

    # LOAD MODEL AND TRAINING HISTORY FROM FILE
    # If specified, continue run by loading Decoder attributes from file (history and model/optim state dicts)
    if resumed_training_file_name is not None:
        load_path = os.getenv('SLURM_SUBMIT_DIR')
        load_path = os.path.join(load_path, 'results/', resumed_training_file_name + '.pt')
        print(('\nLoading training history, weights and optimizer to resume training '
            f'from {load_path}'))
        device = torch.device('cuda')
        current_device_id = torch.cuda.current_device()
        loaded_attributes = torch.load(load_path, map_location=f'cuda:{current_device_id}')
        decoder.load_training_history(loaded_attributes)
        decoder.model.to(device)
        
    # TRAIN
    print('\n==== TRAINING ====')
    decoder.train_with_data_buffer(
        code_size = code_size,
        repetitions = repetitions,
        error_rate = training_error_rate,
        train = True,
        save_to_file = True,
        batch_size = batch_size,
        learning_rate = learning_rate,
        num_iterations = num_iterations,
        benchmark = benchmark,
        buffer_size = buffer_size,
        replacements_per_iteration = replacements_per_iteration,
        test_size = test_size,
        learning_scheduler = False,
        validation = validation)

    print('\n==== TESTING ====')
    rates = [0.001, 0.002, 0.003]
    for r in rates:
        acc = decoder.train_with_data_buffer(
            code_size = code_size,
            repetitions = repetitions,
            error_rate = r,
            train = False,
            test_size = test_size)
        print(f'Test accuracy: {acc}, Error rate: {r}')