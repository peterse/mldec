"""Training loop for decoding experiments with several cycles (repetitions) of measurement."""

import torch
from torch_geometric.loader import DataLoader
import copy
import time

from mldec.utils import evaluation, training
from mldec.models import initialize, baselines
from mldec.pipelines import loggingx


def train_model(model_wrapper, dataset_module_str, config, validation_dataset_config, knob_settings, manager=None):

    # de-serializing the dataset module (for pickling purposes)
    if dataset_module_str == "reps_toric_code":
        from mldec.datasets import reps_toric_code_data
        dataset_module = reps_toric_code_data
    elif dataset_module_str == "reps_exp_rep_code":
        from mldec.datasets import reps_exp_rep_code_data
        dataset_module = reps_exp_rep_code_data

    device = torch.device(config.get('device'))
    if manager is not None:
        log_print = manager.log_print
        job_id = manager.job_id
    else:
        logger = loggingx.get_logger(config.get('log_name'))
        log_print = logger.info
        job_id = None

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    n = config['n']
    mode = config['mode']
    n_train = int(config['n_train'])
    n_test = config['n_test']

    # dump the hyperparameters
    log_print(f"Hyperparameters:")
    for k, v in config.items():
        log_print(f"  {k}: {v}")

    history = []    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer, scheduler = training.initialize_optimizer(config, model_wrapper.model.parameters())
    early_stopping = training.EarlyStopping(patience=patience) 
    max_val_acc = -1

    tot_params, trainable_params = initialize.count_parameters(model_wrapper.model)
    log_print(f"Training model {config.get('model')} with {tot_params} total parameters, {trainable_params} trainable.")


    # TURNING THE KNOB: We may use different dataset configs for training vs. validation
    training_dataset_config = {k: v for k, v in validation_dataset_config.items()}
    training_dataset_config.update(knob_settings) # overwrite the validation dataset using knob settings
    # dump the validation and training dataset configs
    log_print(f"Validation dataset config:")
    for k, v in validation_dataset_config.items():
        log_print(f"  {k}: {v}")
    log_print(f"Training dataset config:")
    for k, v in training_dataset_config.items():
        log_print(f"  {k}: {v}")

    # n_batches = n_train // batch_size
    if dataset_module_str == "reps_toric_code":
        # This is the stim simulated experiments path
        data_tr, triv_tr, _, _ = dataset_module.sample_dataset(n_train, training_dataset_config, device)
        data_val, triv_val, stim_data_val, observable_flips_val = dataset_module.sample_dataset(n_test, validation_dataset_config, device)
    elif dataset_module_str == "reps_exp_rep_code":
        # this is for (simulated) experimental data that was already generated.
        data_tr, triv_tr, _ = dataset_module.sample_dataset(n_train, training_dataset_config, device)
        # the validation data from experiments is coded as beta=0
        assert validation_dataset_config.get("beta") == 0, "Validation dataset config must have beta=0 for experimental data"
        data_val, triv_val, observable_flips_val = dataset_module.sample_dataset(n_test, validation_dataset_config, device)
    else:
        raise ValueError("Unknown dataset module")
    
    log_print("generated data")
    log_print("number of trivial training samples: {}".format(triv_tr))
    log_print("number of trivial validation samples: {}".format(triv_val))
    train_dataloader = DataLoader(data_tr, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(data_val, batch_size=n_test, shuffle=False)

    # Baseline accuracies: set up pymatching decoder for validation set; do this directly on stim detection events
    if dataset_module_str == "reps_toric_code":
        _, _, detector_error_model = dataset_module.make_sampler(validation_dataset_config)
        mwpm_decoder = baselines.CyclesMinimumWeightPerfectMatching(detector_error_model)
        minimum_weight_correct_nontrivial = evaluation.evaluate_mwpm(stim_data_val, observable_flips_val, mwpm_decoder).item()
        minimum_weight_val_acc = (minimum_weight_correct_nontrivial + triv_val) / n_test
        log_print("minweight acc: {}".format(minimum_weight_val_acc))
    elif dataset_module_str == "reps_exp_rep_code":
        # FIXME: implement this
        minimum_weight_val_acc = 999
        log_print("minweight acc: {}".format(minimum_weight_val_acc))

    # training loop for model begins:
    t_start = time.time()
    for epoch in range(max_epochs):
        train_loss = 0      
        correct_nontrivial_preds_tr = 0
        for data_batch in train_dataloader: # batched training
            correct_nontrivial_preds_tr_batch, loss = model_wrapper.training_step(data_batch, optimizer, criterion)
            train_loss += loss.item()
            correct_nontrivial_preds_tr += correct_nontrivial_preds_tr_batch
        train_loss /= len(train_dataloader) # average loss over batches

        # Compute accuracies as (correct predictions + trivial count) / (n_data + trivial count)
        train_acc = (correct_nontrivial_preds_tr + triv_tr) / n_train
        if (epoch % 10) == 0:
            # Compute accuracies: Happens every 10 epochs
            model_wrapper.model.eval()        
            # correct_nontrivial_preds_tr = evaluation.batched_correct_predictions(data_tr, model_wrapper.model, device)
            correct_nontrivial_preds_val = evaluation.batched_correct_predictions(val_dataloader, model_wrapper.model, device)
            val_acc = (correct_nontrivial_preds_val + triv_val) / n_test
            epoch_results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_loss": 0, # val_loss is not computed
                    "vs_lookup": 0, # TODO
                    "vs_minweight": val_acc - minimum_weight_val_acc
                }
            history.append(epoch_results)
            save_str = ""
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_results = copy.deepcopy(epoch_results)
            time_elapsed = time.time() - t_start
            log_print(f"Epoch {epoch+1}/{max_epochs} | Time elapsed: {time_elapsed:4.1f} | Train Loss: {train_loss:.4E} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}" + save_str)

            if config["mode"] == 'tune':
                manager.report(epoch_results)

        early_stopping(train_loss, model_wrapper.model)
        if early_stopping.early_stop:
            log_print("Early stopping")
            break

        if epoch == max_epochs - 1:
            log_print("Max epochs reached")

    auxiliary_results = {
        "job_id": job_id,
        "total_parameters": tot_params,
        "total_time": time.time() - t_start,
        "total_epochs": epoch,
    }
    best_results.update(auxiliary_results)
    # return the final results
    return best_results



