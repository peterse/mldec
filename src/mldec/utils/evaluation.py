import torch
from torch import nn


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
        loss = self.crit(inputs.view(total_sz), targets.view(total_sz)).view(batchsz, -1)  # [B, T]
        loss = torch.dot(self._reduce(loss), weight.type_as(loss))
        return loss

    def extra_repr(self):
        return f"reduction={self.avg}"
    

def weighted_accuracy(model, X, Y, weights):
    Y_pred = model(X).int() # get predictions from activations
    compare = ((Y_pred + Y) % 2).sum(axis=1) == 0
    acc = (compare * weights).sum()
    return acc.item()


def weighted_loss(model, X, Y, weights, criterion):
    Y_pred = model(X)
    res = criterion(Y_pred, Y, weights)
    return res.item()
    