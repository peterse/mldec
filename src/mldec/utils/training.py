import torch
from torch import nn, optim

class EarlyStopping:
    def __init__(self, patience=5, delta=0, logger=None):
        """Implement early stopping if the validation loss doesn't improve after a given patience period.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.logger:
            self.logger.debug(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def initialize_optimizer(config, params):
    lr = config.get('lr')
    opt = config.get('opt')
    decay_rate = config.get('sgd_decay_rate')
    decay_patience = config.get('sgd_decay_patience')
    scheduler = None
    if opt == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    elif opt == 'adadelta':
        optimizer = optim.Adadelta(params, lr=lr)
    elif opt == 'asgd':
        optimizer = optim.ASGD(params, lr=lr)
    elif opt =='rmsprop':
        optimizer = optim.RMSprop(params, lr=lr)
    else:
        optimizer = optim.SGD(params, lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=decay_rate, patience=decay_patience, verbose=True)

    return optimizer, scheduler
