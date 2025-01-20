import torch
import numpy as np

from mldec.utils import evaluation, training

def train_model(model, dataset_module, config, dataset_config):

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    patience = config['patience']
    n = config['n']
    mode = config['mode']
    n_train = config['n_train']
    lr = config['lr']
    opt = config['opt']
    decay_rate = config['decay_rate']
    decay_patience = config['decay_patience']
    
    # for epoch in range(max_epochs):
    
    # # FIXME:
    # unique_errors_seen = len(np.unique(Y_train, axis=0))

    # # Train loop
    # model.train()
    # train_loss = 0.0
    # for X_batch, Y_batch in train_loader:
    #     optimizer.zero_grad()
    #     output = model(X_batch)
    #     loss = criterion(output, Y_batch)
    #     loss.backward()
    #     optimizer.step()
    #     train_loss += loss.item()
    # train_loss = train_loss / len(train_loader)

    # # Validation: 'validation' means evaluating the weighted loss on the true distribtion
    # # a model trained in this way cannot be any _worse_ than a model without that validiation scheme!
    # # WITH RESPECT TO ORIGINAL PROBABILITIES
    # model.eval()
    # train_acc = evaluate_model(model, X_train_tensor, Y_train_tensor, print_results=False)
    # if (epoch % 100) == 0:
    #     print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.8f}  | Train Acc: {train_acc:.4f} ")

    # model = FFNNlayered(input_dim=n-1, hidden_dim=HIDDEN_DIM, output_dim=n, N_layers=N_LAYERS)
    # print("model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    criterion = evaluation.WeightedSequenceLoss(torch.nn.BCELoss )

    opt = config.get('opt')
    optimizer = training.initialize_optimizer(config, model.parameters())
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
        X, Y = dataset_module.create_dataset(n)
        
        weights_tensor = torch.tensor(dataset_module.noise_model(Y, n, dataset_config), dtype=torch.float32)  # true distribution of bitstrings  
        train_loss = 0
        all_batches = []
        for _ in range(n_batches):
            # Sample a virtual batch of data
            # CAREFUL: the shape of weights varies with each batch; histb is consistently (2**nbits,)
            Xb, Yb, weightsb, histb = dataset_module.sample_virtual_XY(weights_tensor.numpy(), batch_size, n, dataset_config)
            downsampled_weights += histb
            all_batches.append((Xb, Yb, weightsb))
        
        for _, batch in enumerate(all_batches):
            # Do gradient descent on a virtual batch of data
            Xb, Yb, weightsb = batch
            optimizer.zero_grad()
            Y_pred = model(Xb)
            loss = criterion(Y_pred, Yb, weightsb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / n_batches
        downsampled_weights /= n_batches

        # Virtual Validation
        downsampled_weights_tensor = torch.tensor(downsampled_weights, dtype=torch.float32)
        model.eval()        
        val_loss = evaluation.weighted_loss(model, X, Y, weights_tensor, criterion)        
        val_acc = evaluation.weighted_accuracy(model, X, Y, weights_tensor)
        train_acc = evaluation.weighted_accuracy(model, X, Y, downsampled_weights_tensor) # training accuracy is evaluated on the same data from this epoch.

        if config.get('opt') == 'sgd':
            model.scheduler.step(val_acc)

        if config["mode"] == 'tune':
            train.report({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_loss,
            })

        if (epoch % 10) == 0:
            # Saving and printing
            save_str = ""
            if val_acc > max_val_acc:
                # torch.save(model.state_dict(), 'checkpoint.pt')
                max_val_acc = val_acc
                # save_str = " (Saved)"
            print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}" + save_str)

        history.get("train_loss").append(train_loss)
        history.get("val_loss").append(val_loss)
        history.get("train_acc").append(train_acc)
        history.get("val_acc").append(val_acc)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # raise GetOutOfLoop
            break

        if epoch == max_epochs - 1:
            print("Max epochs reached")



    
def train_model_group(hyper_settings, hyper_config, train_loader, val_loader, noiseless_val_loader, voc, 
				config, logger, epoch_offset= 0):	
	
	ray.init(include_dashboard=False) # suppress dashboard resources

	# config should have tune=True
	scheduler = ASHAScheduler(
		time_attr="training_iteration",
		metric="val_acc",
		mode="max",
		max_t=hyper_settings.get("max_iterations"), # I'm not sure what this kwarg does and neither is the documentation
		grace_period=hyper_settings.get("grace_period"), 
		reduction_factor=2)

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
		# stop={"training_iteration": config.epochs, "val_acc": 0.80},
	)
	# # pdb.set_trace()
	# for v in [model, train_loader, val_loader, voc, device, config, logger]:
	# 	print("checking object", v)
	# 	print(inspect_serializability(v))
	# 	print()

	resources = tune.with_resources(
				tune.with_parameters(
					build_and_train_model_raytune, 
					config=config,
					train_loader=train_loader,
					val_loader=val_loader,
					noiseless_val_loader=noiseless_val_loader,
					voc=voc,
					logger=logger,
					epoch_offset=epoch_offset,
					),
				resources={"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")}
	)

	tuner = Tuner(
		resources,
		param_space=hyper_config,
		tune_config=tune_config,
		run_config=run_config,
	)
	result = tuner.fit()

	return result
