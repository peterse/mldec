import torch
import numpy as np
import copy

from mldec.utils import evaluation, training
from mldec.models import initialize, baselines
from mldec.pipelines import loggingx


def train_model(model_wrapper, dataset_module, config, validation_dataset_config, knob_settings, manager=None):

    # de-serializing the dataset module
    if dataset_module == "reps_toric_code":
        from mldec.datasets import reps_toric_code_data
        dataset_module = reps_toric_code_data

    device = torch.device(config.get('device'))
    if manager is not None:
        log_print = manager.log_print
    else:
        logger = loggingx.get_logger(config.get('log_name'))
        log_print = logger.info

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    n = config['n']
    mode = config['mode']
    n_train = config['n_train']

    # dump the hyperparameters
    log_print(f"Hyperparameters:")
    for k, v in config.items():
        log_print(f"  {k}: {v}")

    history = []    
    criterion = torch.nn.BCEWithLogitsLoss
    optimizer, scheduler = training.initialize_optimizer(config, model_wrapper.model.parameters())
    early_stopping = training.EarlyStopping(patience=patience) 
    max_val_acc = -1

    tot_params, trainable_params = initialize.count_parameters(model_wrapper.model)
    log_print(f"Training model {config.get('model')} with {tot_params} total parameters, {trainable_params} trainable.")


    # TURNING THE KNOB: We will use (potentially) different dataset configs for training vs. validation
    # validation_dataset_config = copy.deepcopy(validation_dataset_config)
    training_dataset_config = {}
    for k, v in validation_dataset_config.items():
        if knob_settings.get(k) is not None:
            training_dataset_config[k] = knob_settings[k] # overwrite the validation dataset using knob settings
        else:
            training_dataset_config[k] = v # use the same as the validation set
    # dump the validation and training dataset configs
    log_print(f"Validation dataset config:")
    for k, v in validation_dataset_config.items():
        log_print(f"  {k}: {v}")
    log_print(f"Training dataset config:")
    for k, v in training_dataset_config.items():
        log_print(f"  {k}: {v}")


    n_batches = n_train // batch_size
    # compute the probability weights corresponding to the p for this distribtion
    X_val, Y_val = dataset_module.sample_dataset(n, validation_dataset_config)
    # compute the probability weights after 'turning the knob' on the p
    X_tr, Y_tr = dataset_module.sample_dataset(n, training_dataset_config)
    # copy the weights
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)








    for data in loader:
        model.training_step(data, optimizer, criterion)



    #   def train_with_buffer(graph_list, shuffle=True):
    #         '''Trains the network with data from the buffer.'''
    #         loader = DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)
    #         total_loss = 0.
    #         correct_predictions = 0
    #         model.train()




    #             total_loss += loss.item() * data.num_graphs

    #         return correct_predictions, total_loss
        


    batched_data = []

    # build a train set one batch at a time
    # this will accumulate a histogram of the training set over all batches
    downsampled_train_weights = np.zeros_like(train_weights_np)
    for _ in range(n_batches):
        Xb, Yb, train_weightsb, histb = dataset_module.sample_virtual_XY(train_weights_np, batch_size, n, training_dataset_config)
        Xb, Yb, train_weightsb = Xb.to(device), Yb.to(device), train_weightsb.to(device)
        downsampled_train_weights += histb
        batched_data.append((Xb, Yb, train_weightsb))
    downsampled_train_weights /= n_batches # histogram of the training set

    log_print("Train weights:")
    log_print(downsampled_train_weights)

    # we want two things: a lookup table for the training set, and baseline accuracy for training/validation

    # Baseline accuracies
    lookup_decoder = baselines.LookupTable()
    if config.get("dataset_module") == "toy_problem":
        minimum_weight_decoder = baselines.RepetitionCodeMinimumWeight()
    elif config.get("dataset_module") == "toric_code":
        minimum_weight_decoder = baselines.MinimumWeightPerfectMatching()
    else:
        raise ValueError("Unknown dataset module")

    # TODO: BASELINES
    # lookup_decoder.train_on_histogram(X, Y_no_sos_eos, downsampled_train_weights)
    # lookup_val_acc = evaluation.weighted_accuracy(lookup_decoder, X, Y_no_sos_eos, val_weights)
    # log_print("lookup acc: {}".format(lookup_val_acc))
    # minimum_weight_decoder.make_decoder(X, Y_no_sos_eos)
    # minimum_weight_val_acc = evaluation.weighted_accuracy(minimum_weight_decoder, X, Y_no_sos_eos, val_weights)
    # log_print("minweight acc: {}".format(minimum_weight_val_acc))

    # We will keep the best results (according to val acc) and return only those.
    best_results = None
    for epoch in range(max_epochs):


        train_loss = 0        
        for i in range(n_batches):
            Xb, Yb, weightsb = batched_data[i]
            # Do gradient descent on a virtual batch of data
            loss = model_wrapper.training_step(Xb, Yb, weightsb, optimizer, criterion)
            train_loss += loss.item()
        train_loss = train_loss / n_batches


        # forward pass:
        correct_count, total_iteration_loss = train_with_buffer(data_buffer)
        sample_count = len(data_buffer)
        # update the data buffer (either replace or append batches)
        if replacements_per_iteration > 0:
            data_buffer = update_buffer(data_buffer, replacements_per_iteration)
        gc.collect()
        # Store loss and accuracy from training iteration
        train_loss = total_iteration_loss / sample_count
        train_acc = correct_count / sample_count
        # If validation, test on validation batch



        

        # We only do accuracy and loss checks every 10 epochs
        if (epoch % 10) == 0:
            # Virtual Validation: Happens every 10 epochs; on whatever dataset we get.
            downsampled_train_weights_tensor = torch.tensor(downsampled_train_weights, dtype=torch.float32).to(device)
            model_wrapper.model.eval()        
            # Heads up: For autoregressive models, train_loss tracks something very different than val_loss!
            # val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
        
            # # # # # # # # val ac computation

            # Initialize data buffer with train/test split
            # I DON'T REALLY NEED THIS.
            # data_buffer = generate_buffer(buffer_size + test_size)
            # test_val_batch = data_buffer[:(test_size * batch_size * len(error_rate))]
            # data_buffer = data_buffer[(test_size * batch_size * len(error_rate)):]
            # gc.collect()
            # val_acc = decode_test_batch(test_val_batch)



            # # # # # # # # train acc computation
            train_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, downsampled_train_weights_tensor)



            # # # # # test acc calculation
            # Generate a test batch
            graph_batch, correct_predictions_trivial = generate_test_batch(test_size)
            loader = DataLoader(graph_batch, batch_size = 1000)
            correct_predictions = 0
            self.model.eval()            # run network in training mode 
            with torch.no_grad():   # turn off gradient computation (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
                for data in loader:
                    # Perform forward pass to get network output
                    prediction = self.predict(data)
                    target = data.y.to(int) # Assumes binary targets (no probabilities)
                    correct_predictions += int( (prediction == target).sum() )

            # Count correct predictions by GNN for nontrivial syndromes
            val_acc = (correct_predictions + correct_predictions_trivial) / test_size

            epoch_results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "vs_lookup": val_acc - lookup_val_acc,
                    "vs_minweight": val_acc - minimum_weight_val_acc
                }
            history.append(epoch_results)
            save_str = ""
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_results = copy.deepcopy(epoch_results)
            log_print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4E} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4E} | Val Acc: {val_acc:.4f}" + save_str)

            if config["mode"] == 'tune':
                manager.report(epoch_results)

        early_stopping(val_loss, model_wrapper.model)
        if early_stopping.early_stop:
            log_print("Early stopping")
            # raise GetOutOfLoop
            break

        if mode == 'verify' and np.allclose(val_acc, 1.0) and np.allclose(train_acc, 1.0):
            log_print("Early stopping: perfect accuracy")
            log_print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4E} | Val Loss: {val_loss:.4E} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}" + save_str)
            break

        if epoch == max_epochs - 1:
            log_print("Max epochs reached")

    # return the final results
    return best_results



