import torch
import numpy as np
import os
import datetime

from mldec.utils import evaluation, training
from mldec.models import initialize
from ray.train.torch import get_device as raytune_get_device


import ray
from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig
from ray.tune.schedulers import FIFOScheduler


from mldec.utils import evaluation, training
from mldec.models import initialize

def train_model(model_wrapper, dataset_module, config, dataset_config):

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
    print(f"Training model {config.get('model')} with {tot_params} total parameters, {trainable_params} trainable.")

    # Training loop
    for epoch in range(max_epochs):

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
        X, Y, weights = X.to(config.get('device')), Y.to(config.get('device')), weights.to(config.get('device'))

        train_loss = 0        
        for _ in range(n_batches):
            Xb, Yb, weightsb, histb = dataset_module.sample_virtual_XY(weights_np, batch_size, n, dataset_config)
            Xb, Yb, weightsb = Xb.to(config.get('device')), Yb.to(config.get('device')), weightsb.to(config.get('device'))
            downsampled_weights += histb
            # Do gradient descent on a virtual batch of data
            loss = model_wrapper.training_step(Xb, Yb, weightsb, optimizer, criterion)
            train_loss += loss.item()

        train_loss = train_loss / n_batches
        downsampled_weights /= n_batches

        # if config.get('opt') == 'sgd':
        #     scheduler.step(val_acc)

        # We only do accuracy and loss checks every 10 epochs
        if (epoch % 100) == 0:
            # Virtual Validation: Happens every 10 epochs; on whatever dataset we get.
            downsampled_weights_tensor = torch.tensor(downsampled_weights, dtype=torch.float32).to(config.get('device'))
            model_wrapper.model.eval()        
            # Heads up: For autoregressive models, train_loss tracks something very different than val_loss!
            # val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
            if config.get('model') == 'encdec':
                val_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, weights)
                val_preds = model_wrapper.model(X, Y[:,:-1], tgt_mask=model_wrapper.tgt_mask)
                val_loss = criterion(val_preds, Y[:, 1:], weights).item()
            else:
                raise(NotImplementedError)
                val_acc, val_loss = evaluation.weighted_accuracy_and_loss(model_wrapper, X, Y, weights, criterion)
            train_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, downsampled_weights_tensor) # training accuracy is evaluated on the same data from this epoch.

            # DEBUG
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
            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4E} | Val Loss: {val_loss:.4E} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}" + save_str)
            
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
                ray.train.report(epoch_results)

        early_stopping(val_loss, model_wrapper.model)
        if early_stopping.early_stop:
            print("Early stopping")
            # raise GetOutOfLoop
            break

        if mode == 'verify' and np.allclose(val_acc, 1.0) and np.allclose(train_acc, 1.0):
            print("Early stopping: perfect accuracy")
            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4E} | Val Loss: {val_loss:.4E} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}" + save_str)
            break

        if epoch == max_epochs - 1:
            print("Max epochs reached")



def tune_hyperparameters(hyper_config, hyper_settings, dataset_module, config, dataset_config):


    dataset = config.get("dataset_module")
    only_good_examples = config.get("only_good_examples")
    prefix = f"{dataset}_"
    if only_good_examples:
        prefix += "only_good_examples_"
    prefix += f"{config.get('model')}_"
    # create a MMDDHHMMSS timestamp
    suffix = datetime.datetime.now().strftime("%m%d%H%M%S")

    def trial_dirname_creator(trial):
        return f"{prefix}{trial.trial_id}"

    ray.init(
        include_dashboard=False, 
            num_cpus=hyper_settings.get("total_cpus"), 
            num_gpus=hyper_settings.get("total_gpus"), 
            _temp_dir=None, 
            ignore_reinit_error=True)
    
    scheduler = FIFOScheduler()

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration", "mean_accuracy"],
        print_intermediate_tables=False,
        )

    tune_config = tune.TuneConfig(
        num_samples=hyper_settings.get("num_samples"),
        scheduler=scheduler,
        trial_dirname_creator=trial_dirname_creator,
        max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
        )
        
    run_config = RunConfig(
        progress_reporter=reporter,
        storage_path=hyper_settings.get("tune_directory"),
        stop={"epoch": hyper_config.get("max_epochs")}, # because both the trainable and raytune need to see max_epochs.
    )

    resources = tune.with_resources(
                tune.with_parameters(
                    initialize_and_train_model, 
                    config=config,
                    dataset_module=dataset_module,
                    dataset_config=dataset_config,
                    ),
                resources={"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")}
    )

    tuner = tune.Tuner(
        resources,
        param_space=hyper_config,
        tune_config=tune_config,
        run_config=run_config,
    )
    result = tuner.fit()
    df = result.get_dataframe()
    dest = os.path.join(hyper_settings.get("tune_directory"), f"{prefix}{suffix}_tune.csv")
    with open(dest, 'a') as f:
        f.write(df.to_string(header=True, index=False))
        
    return result


def initialize_and_train_model(hyper_config, config, dataset_module, dataset_config):

    device = raytune_get_device()
    print("initializing raytune device:", device)
    config["device"] = device
    # merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
    for k, v in hyper_config.items():
        config[k] = v
    model = initialize.initialize_model(config)
    train_model(model, dataset_module, config, dataset_config)
