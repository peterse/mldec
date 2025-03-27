import torch
import numpy as np
import copy

from mldec.utils import evaluation, training
from mldec.models import initialize, baselines
from mldec.pipelines import loggingx

def train_model(model_wrapper, dataset_module, config, validation_dataset_config, knob_settings, manager=None):

    # de-serializing the dataset module
    if dataset_module == "toy_problem":
        from mldec.datasets import toy_problem_data
        dataset_module = toy_problem_data
    elif dataset_module == "toric_code":
        from mldec.datasets import toric_code_data
        dataset_module = toric_code_data

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
    criterion = evaluation.WeightedSequenceLoss(torch.nn.BCEWithLogitsLoss)

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
    # Virtual TRAINING: We don't actually need datasets to train the model. Instead,
    # we sample data ccording to the known probability distribution, and then reweight the loss
    # according to a histogram of that sample. The loss re-weighting is O(2^nbits) so this is
    # more efficient whenever that number is much smaller than the expected amount of training data.
    # TODO: optimization for bandwidth; maybe pre-load large chunks of data since its only a few KB
    # per batch
    n_batches = n_train // batch_size
    if config.get('only_good_examples'):
        X, Y, val_weights = dataset_module.uniform_over_good_examples(n, validation_dataset_config)
        train_weights = val_weights # no knob in this setting
    else:
        # compute the probability weights corresponding to the p for this distribtion
        X, Y, val_weights = dataset_module.create_dataset_training(n, validation_dataset_config)
        # compute the probability weights after 'turning the knob' on the p
        _, _, train_weights = dataset_module.create_dataset_training(n, training_dataset_config)
    # copy the weights
    train_weights_np = train_weights.numpy()
    val_weights = torch.tensor(val_weights, dtype=torch.float32)  # true distribution of bitstrings  
    X, Y, val_weights = X.to(device), Y.to(device), val_weights.to(device)
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
    log_print("Test val_weights:")
    log_print(val_weights)
    if config.get("dataset_module") == "toric_code":
        log_print("Probability of no error in training error model:")
        log_print(toric_code_data.make_variance_noise_model(n, training_dataset_config)(np.zeros(2*n), n))

    # we want two things: a lookup table for the training set, and baseline accuracy for training/validation

    # Baseline accuracies
    lookup_decoder = baselines.LookupTable()
    if config.get("dataset_module") == "toy_problem":
        minimum_weight_decoder = baselines.RepetitionCodeMinimumWeight()
    elif config.get("dataset_module") == "toric_code":
        minimum_weight_decoder = baselines.MinimumWeightPerfectMatching()
    else:
        raise ValueError("Unknown dataset module")
    # we need to strip the EOS/SOS from the training data, if applicable.
    if validation_dataset_config.get("sos_eos") is not None:
        Y_no_sos_eos = torch.clone(Y)[:, 1:-1]
    else:
        Y_no_sos_eos = torch.clone(Y)
    lookup_decoder.train_on_histogram(X, Y_no_sos_eos, downsampled_train_weights)
    lookup_val_acc = evaluation.weighted_accuracy(lookup_decoder, X, Y_no_sos_eos, val_weights)
    log_print("lookup acc: {}".format(lookup_val_acc))
    minimum_weight_decoder.make_decoder(X, Y_no_sos_eos)
    minimum_weight_val_acc = evaluation.weighted_accuracy(minimum_weight_decoder, X, Y_no_sos_eos, val_weights)
    log_print("minweight acc: {}".format(minimum_weight_val_acc))

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


        # if config.get('opt') == 'sgd':
        #     scheduler.step(val_acc)

        # We only do accuracy and loss checks every 10 epochs
        if (epoch % 10) == 0:
            # Virtual Validation: Happens every 10 epochs; on whatever dataset we get.
            downsampled_train_weights_tensor = torch.tensor(downsampled_train_weights, dtype=torch.float32).to(device)
            model_wrapper.model.eval()        
            # Heads up: For autoregressive models, train_loss tracks something very different than val_loss!
            # val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
            if config.get('model') == 'transformer':
                val_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, val_weights)
                val_preds = model_wrapper.model(X, Y[:,:-1], tgt_mask=model_wrapper.tgt_mask)
                val_loss = criterion(val_preds, Y[:, 1:], val_weights).item()
            else:
                val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, val_weights, criterion)
            train_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, downsampled_train_weights_tensor)

            # DEBUG for encdec
            # tgt_input = Y[:, :-1] # shape [batch, output_len - 1]
            # tgt_out = Y[:, 1:] # shape [batch, output_len - 1]
            # model_out = model_wrapper.model(X, tgt_input)
            # pred_out = (model_out >= 0).long()
            # print("training comparison")
            # for y, ypred, w in zip(tgt_out, pred_out, downsampled_train_weights):
            #     if w > 0:
            #         print(y, ypred)
            # print()

            # print("prediction comparison:")
            # Y_pred = model_wrapper.predict(X)
            # for y, ypred, w in zip(Y, Y_pred, weights):
            #     if w > 0:
            #         print(y, ypred)
            # reporting every 10 epochs
            epoch_results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
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
