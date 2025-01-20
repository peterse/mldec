import time

import numpy as np
import torch


def evaluate(model, source, targets):
	
    output = model(source)
    preds = output.cpu().numpy()
    preds = preds.argmax(axis=1)
    labels= targets.cpu().numpy()
    acc= np.array(preds==labels, np.int32).sum() / len(targets)
    return acc


def run_validation(model, data_loader, device):
	"""Evaluate the model's performance on a data set."""
	model.eval()
	batch_num = 0
	val_acc_epoch = 0.0

	with torch.no_grad():
		for batch, i in enumerate(range(0, len(data_loader), data_loader.batch_size)):

			source, targets = data_loader.get_batch(i)
			source, targets = source.to(device), targets.to(device)
			acc = evaluate(model, source, targets)
			val_acc_epoch += acc
			batch_num += 1

	val_acc_epoch = val_acc_epoch/data_loader.num_batches

	return val_acc_epoch



def train_model(model, train_loader, val_loader, voc, device, 
				config, logger, epoch_offset= 0, min_val_loss=1e7, 
				max_val_acc=0.0):

	best_epoch = 0
	curr_train_acc=0.0
	early_stop_count=0
	max_train_acc = 0.0

	itr= 0
	gen_success=False
	conv_time = -1
	conv = False
	data_size = len(train_loader)
	estop_lim = 1000 * (config.batch_size // data_size)

	for epoch in range(1, config.epochs):

		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		val_acc_epoch = 0.0
		model.train()
		start_time = time()
		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']

		for batch, i in enumerate(range(0, len(train_loader), config.batch_size)):

			# if config.model_type == 'RNN':
			# 	hidden = model.model.init_hidden(config.batch_size)
			# else:
			# 	hidden = None
		
			source, targets, word_lens = train_loader.get_batch(i)			
			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)
			loss = model.trainer(source, targets, word_lens, config)
			train_loss_epoch += loss 
			itr +=1
		
		train_loss_epoch = train_loss_epoch/train_loader.num_batches
		time_taken = time() - start_time
		time_mins = int(time_taken/60)
		time_secs= time_taken%60

		logger.debug('Training for epoch {} completed...\nTime Taken: {} mins and {} secs'.format(epoch, time_mins, time_secs))
		logger.debug('Starting Validation')

		val_acc_epoch = run_validation(config, model, val_loader, voc, device, logger)
		train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)
		gen_gap = train_acc_epoch - val_acc_epoch

		if config.opt == 'sgd':
			model.scheduler.step(val_acc_epoch)

		if config.wandb:
			wandb.log({
				'train-loss': train_loss_epoch,
				'train-acc': train_acc_epoch,
				'val-acc':val_acc_epoch,
				'gen-gap': gen_gap,
				})

		if config.mode == 'tune':
			train.report({
				"train_loss": (train_loss_epoch),
				"train_acc": train_acc_epoch,
				"val_acc": val_acc_epoch
			})

		if val_acc_epoch > max_val_acc :
			max_val_acc = val_acc_epoch
			best_epoch= epoch
			curr_train_acc= train_acc_epoch

		if val_acc_epoch> 0.9999:
			early_stop_count +=1

		else:
			early_stop_count=0

		if early_stop_count > estop_lim:
			break
		if train_acc_epoch> 0.999:
			early_stop_tr +=1
		else:
			early_stop_tr=0

		if early_stop_tr > 30:
			break

		
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['train_loss'] = train_loss_epoch
		od['train_acc'] = train_acc_epoch
		od['val_acc_epoch']= val_acc_epoch
		od['max_val_acc']= max_val_acc
		od['lr_epoch'] = lr_epoch
		# od['conv_time'] = conv_time
		print_log(logger, od)

	logger.info('Training Completed for {} epochs'.format(config.epochs))


	if config.results:
		store_results(config, max_val_acc, curr_train_acc, best_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))
