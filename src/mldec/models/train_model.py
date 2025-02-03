import torch
import numpy as np
import os
import datetime
import json
import time
import multiprocessing as mp

from mldec.utils import evaluation, training
from mldec.models import initialize
from mldec.pipelines import loggingx

def train_model(model_wrapper, dataset_module, config, dataset_config, manager=None):

    # de-serializing
    if dataset_module == "toy_problem":
        from mldec.datasets import toy_problem_data
        dataset_module = toy_problem_data
    device = torch.device(config.get('device'))
    if manager is not None:
        log_print = manager.log_print
    else:
        logger = loggingx.init_logger("train_model")
        log_print = logger.info

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    n = config['n']
    mode = config['mode']
    n_train = config['n_train']

    history = []    
    criterion = evaluation.WeightedSequenceLoss(torch.nn.BCEWithLogitsLoss)

    optimizer, scheduler = training.initialize_optimizer(config, model_wrapper.model.parameters())
    early_stopping = training.EarlyStopping(patience=patience) 
    max_val_acc = -1

    tot_params, trainable_params = initialize.count_parameters(model_wrapper.model)
    log_print(f"Training model {config.get('model')} with {tot_params} total parameters, {trainable_params} trainable.")


    # Virtual TRAINING: We don't actually need datasets to train the model. Instead,
    # we sample data ccording to the known probability distribution, and then reweight the loss
    # according to a histogram of that sample. The loss re-weighting is O(2^nbits) so this is
    # more efficient whenever that number is much smaller than the expected amount of training data.
    # TODO: optimization for bandwidth; maybe pre-load large chunks of data since its only a few KB
    # per batch
    n_batches = n_train // batch_size
    downsampled_weights = np.zeros(2**n) # this will accumulate a histogram of the training set over all batches
    if config.get('only_good_examples'):
        X, Y, weights = dataset_module.uniform_over_good_examples(n, dataset_config)
    else:
        X, Y, weights = dataset_module.create_dataset_training(n, dataset_config)
    # copy the weights
    weights_np = weights.numpy()
    weights = torch.tensor(weights, dtype=torch.float32)  # true distribution of bitstrings  
    X, Y, weights = X.to(device), Y.to(device), weights.to(device)
    batched_data = []

    # build a train set one batch at a time
    for _ in range(n_batches):
        Xb, Yb, weightsb, histb = dataset_module.sample_virtual_XY(weights_np, batch_size, n, dataset_config)
        Xb, Yb, weightsb = Xb.to(device), Yb.to(device), weightsb.to(device)
        downsampled_weights += histb
        batched_data.append((Xb, Yb, weightsb))
    downsampled_weights /= n_batches # histogram of the training set

    # Training loop
    for epoch in range(max_epochs):

        train_loss = 0        
        for i in range(n_batches):
            Xb, Yb, weightsb = batched_data[i]
            # Do gradient descent on a virtual batch of data
            loss = model_wrapper.training_step(Xb, Yb, weightsb, optimizer, criterion)
            train_loss += loss.item()
        train_loss = train_loss / n_batches

        # if config.get('opt') == 'sgd':
        #     scheduler.step(val_acc)

        # We only do accuracy and loss checks every 10 epochs
        if (epoch % 10) == 0:
            # Virtual Validation: Happens every 10 epochs; on whatever dataset we get.
            downsampled_weights_tensor = torch.tensor(downsampled_weights, dtype=torch.float32).to(device)
            model_wrapper.model.eval()        
            # Heads up: For autoregressive models, train_loss tracks something very different than val_loss!
            # val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
            if config.get('model') == 'encdec':
                val_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, weights)
                val_preds = model_wrapper.model(X, Y[:,:-1], tgt_mask=model_wrapper.tgt_mask)
                val_loss = criterion(val_preds, Y[:, 1:], weights).item()
            else:
                val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
            train_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, downsampled_weights_tensor) # training accuracy is evaluated on the same data from this epoch.

            # DEBUG for encdec
            # tgt_input = Y[:, :-1] # shape [batch, output_len - 1]
            # tgt_out = Y[:, 1:] # shape [batch, output_len - 1]
            # model_out = model_wrapper.model(X, tgt_input)
            # pred_out = (model_out >= 0).long()
            # print("training comparison")
            # for y, ypred, w in zip(tgt_out, pred_out, downsampled_weights):
            #     if w > 0:
            #         print(y, ypred)
            # print()

            # print("prediction comparison:")
            # Y_pred = model_wrapper.predict(X)
            # for y, ypred, w in zip(Y, Y_pred, weights):
            #     if w > 0:
            #         print(y, ypred)

            # Saving and printing
            save_str = ""
            if val_acc > max_val_acc:
                # torch.save(model.state_dict(), 'checkpoint.pt')
                max_val_acc = val_acc
                # save_str = " (Saved)"
            results = [epoch, train_loss, train_acc, val_loss, val_acc]
            log_print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4E} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4E} | Val Acc: {val_acc:.4f}" + save_str)
            
            # reporting every 10 epochs
            epoch_results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            history.append(epoch_results)

            if config["mode"] == 'tune':
                # ray.train.report(epoch_results)
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
    return results
