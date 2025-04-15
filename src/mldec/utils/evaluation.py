import torch
from torch import nn
import numpy as np


class WeightedSequenceLoss(nn.Module):
    """Weight individual training examples

    """
    def __init__(self, LossFn: nn.Module = nn.NLLLoss, avg: str = "token"):
        super().__init__()
        self.avg = avg
        self.crit = LossFn(reduction="none")
        if avg == 'token':
            self._reduce = self._mean
        else:
            self._reduce = self._sum

    def _mean(self, loss):
        return loss.mean(axis=1)

    def _sum(self, loss):
        return loss.sum(axis=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Evaluate some loss over a sequence.
        :param inputs: torch.FloatTensor, [B, T] The scores from the model. Batch First
        :param targets: torch.LongTensor, [B, T] The labels.
        :param weight: sample weights [B, ]
        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        batchsz = weight.shape[0]
        # FIXME: .view doesn't work but .reshape does, this bothers me a bit?
        # https://github.com/cezannec/capsule_net_pytorch/issues/4
        # claim: reshape might copy the tensor :(
        # The problem: `targets` is a slice of the original Y array with shape [B, T-1], but 
        # `targets` keeps the stride of (T, 1) from before. `view` doesn't like this
        loss = self.crit(inputs.view(total_sz), targets.reshape(total_sz)).view(batchsz, -1)  # [B, T]
        loss = torch.dot(self._reduce(loss), weight.type_as(loss))
        return loss

    def extra_repr(self):
        return f"reduction={self.avg}"
    

def weighted_accuracy(model, X, Y, weights):
    """
    model - a Wrapper that exposes a `predict` method
    """
    Y_pred = model.predict(X).int() # get predictions from activations
    compare = ((Y_pred + Y) % 2).sum(axis=1) == 0
    acc = (compare * weights).sum()
    # print("validation_predictions:")
    # for (y, ypred, weight) in zip(Y, Y_pred, weights):
    #     print(y, ypred, weight)
    return acc.item()


def weighted_loss(model, X, Y, weights, criterion):
    """
    model - a Wrapper that exposes a `predict` method
    """
    Y_pred = model.predict(X)
    res = criterion(Y_pred, Y, weights)
    return res.item()

def weighted_accuracy_and_loss(model, X, Y, weights, criterion):
    """
    model - a Wrapper that exposes a `predict` method
    """
    Y_acts = model.predict(X)
    Y_pred = (Y_acts > 0).int()
    loss = criterion(Y_acts, Y, weights)
    compare = ((Y_pred + Y) % 2).sum(axis=1) == 0
    acc = (compare * weights).sum()
    return acc.item(), loss.item()
    

def batched_correct_predictions(data_batch, model, device):
    """For a batch of data, compute the number of correct predictions.
    
    This is specific to the reps_toric_code_data dataset.
    """
    correct_predictions = 0
    with torch.no_grad():
        for data in data_batch:
            data.batch = data.batch.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.y.to(int) # Assumes binary targets (no probabilities)
            # Sum correct predictions
            prediction = torch.sigmoid(out.detach()).round().to(int)
            correct_predictions += int( (prediction == target).sum() )
    return correct_predictions


def evaluate_mwpm(stim_data, observable_flips, model):
    """For a flat list of data, compute the number of correct MWPM predictions."""
    predictions = model.predict(stim_data)
    num_errors = sum(observable_flips ^ predictions)
    # for i in range(len(stim_data)):
    #     actual_for_shot = observable_flips[i]
    #     predicted_for_shot = predictions[i]
    #     # print(actual_for_shot, predicted_for_shot)
    #     # print(actual_for_shot ^ predicted_for_shot)
    #     if (actual_for_shot ^ predicted_for_shot) != 0:
    #         num_errors += 1
    correct_predictions = len(stim_data) - num_errors
    return correct_predictions
