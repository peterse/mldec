import torch
import numpy as np

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
    learning_rate = config['learning_rate']
    patience = config['patience']
    n = config['n']
    mode = config['mode']
    n_train = config['n_train']
    lr = config['lr']
    use_sos_eos = config.get('use_sos_eos', False)

    history = []    
    criterion = evaluation.WeightedSequenceLoss(torch.nn.BCELoss)

    opt = config.get('opt')
    optimizer, scheduler = training.initialize_optimizer(config, model_wrapper.model.parameters())
    early_stopping = training.EarlyStopping(patience=patience) 
    max_val_acc = -1

    # Training loop
    for epoch in range(max_epochs):

        # Virtual TRAINING: We don't actually need datasets to train the model. Instead,
        # we sample data ccording to the known probability distribution, and then reweight the loss
        # according to a histogram of that sample. The loss re-weighting is O(2^nbits) so this is
        # more efficient whenever that number is much smaller than the expected amount of training data.
        n_batches = n_train // batch_size
        downsampled_weights = np.zeros(2**n) # this will accumulate a histogram of the training set over all batches
        if config.get('only_good_examples'):
            X, Y, weights = dataset_module.uniform_over_good_examples(n, dataset_config, use_sos_eos=use_sos_eos)
        else:
            X, Y, weights = dataset_module.create_dataset_training(n, dataset_config, use_sos_eos=use_sos_eos)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)  # true distribution of bitstrings  
        train_loss = 0        

        for _ in range(n_batches):
            Xb, Yb, weightsb, histb = dataset_module.sample_virtual_XY(weights_tensor.numpy(), batch_size, n, dataset_config)
            downsampled_weights += histb
            # Do gradient descent on a virtual batch of data
            loss = model_wrapper.training_step(Xb, Yb, weightsb, optimizer, criterion)
            train_loss += loss.item()

        train_loss = train_loss / n_batches
        downsampled_weights /= n_batches

        # Virtual Validation
        downsampled_weights_tensor = torch.tensor(downsampled_weights, dtype=torch.float32)
        model_wrapper.model.eval()        
        val_loss = evaluation.weighted_loss(model_wrapper, X, Y, weights_tensor, criterion)        
        val_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, weights_tensor)
        train_acc = evaluation.weighted_accuracy(model_wrapper, X, Y, downsampled_weights_tensor) # training accuracy is evaluated on the same data from this epoch.

        if config.get('opt') == 'sgd':
            scheduler.step(val_acc)

        if (epoch % 10) == 0:
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
                    "val_acc": val_loss,
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


def trial_dirname_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def tune_hyperparameters(hyper_config, hyper_settings, dataset_module, config, dataset_config):


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
        stop={"epoch": hyper_settings.get("max_epochs")},
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
    with open(hyper_settings.get("tune_path"), 'a') as f:
        f.write(df.to_string(header=True, index=False))
        
    return result


def initialize_and_train_model(hyper_config, config, dataset_module, dataset_config):

    # merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
    for k, v in hyper_config.items():
        config[k] = v

    model = initialize.initialize_model(config)
    train_model(model, dataset_module, config, dataset_config)
