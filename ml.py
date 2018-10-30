import torch
import torch.nn.functional as F


class PopArt(torch.nn.Module):
    """
    Online target normalization from the paper:
    "Learning values across many orders of magnitude"
    https://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf

    Usage:
    Use the method popart.mse_loss(prediction, target) for calculating the loss.
    This updates normalization statistics, and then returns the differentiable loss.
    >>> popart = PopArt(4)
    >>> p = torch.nn.Parameter(torch.randn(1, 4))
    >>> y = torch.randn(1, 4)
    >>> loss = popart.mse_loss(p, y)
    >>> loss.backward()

    IMPORTANT
    Make sure to include the parameters of PopArt in the optimizer.

    To get predictions that are in same scale as true targets:
    >>> y_hat = popart(p)

    To get normalized predictions:
    >>> y_normalized = popart.normalized(p)
    """

    def __init__(self, num_features, alpha=0.999):
        super(PopArt, self).__init__()
        self.count = 0
        self.α = alpha
        self.β = alpha
        self.register_buffer('m1', torch.zeros(num_features))
        self.register_buffer('m2', torch.ones(num_features))
        self.w = torch.nn.Parameter(torch.ones(num_features))
        self.b = torch.nn.Parameter(torch.zeros(num_features))

    @property
    def µ(self):
        return self.m1

    @property
    def σ(self):
        return torch.abs(self.m2 - self.m1 ** 2) ** 0.5 + 1e-6

    def _update(self, targets):
        n = targets.size(0)
        a = self.α ** n
        b = self.β ** n
        self.m1 = a * self.m1 + (1 - a) * targets.mean(dim=0)
        self.m2 = b * self.m2 + (1 - b) * (targets ** 2).mean(dim=0)

    def mse_loss(self, predictions, targets):
        if self.training:
            self.w.data = self.σ * self.w
            self.b.data = self.σ * self.b + self.µ

            self._update(targets)

            self.w.data = self.w / self.σ
            self.b.data = (self.b - self.µ) / self.σ

        targets_normed = (targets - self.µ) / self.σ
        predictions_normed = self.w * predictions + self.b
        return F.mse_loss(predictions_normed, targets_normed)

    def normalized(self, predictions):
        """
        Returns predictions that are normalized to zero mean unit variance.
        """
        return self.w * predictions + self.b

    def forward(self, predictions):
        """
        Returns the un-normalized (true) prediction values.
        """
        return self.σ * (self.w * predictions + self.b) + self.µ


class WelfordNormalization(torch.nn.Module):

    def __init__(self, num_features):
        """
        Online normalization by estimated mean and variance.
        These statistics are calculated using the following
        online algorithm:

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm

        These statistics are not calculated using a sliding window, but
        change faster in the beginning to later converge. Statistics
        are only updated during training, and not during evaluation.

        Usage:
        >>> norm = WelfordNormalization(4)
        >>> norm.train()
        WelfordNormalization()
        >>> x = torch.randn(10, 4)
        >>> y = norm(x)
        """
        super(WelfordNormalization, self).__init__()
        self._count = 0
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('variance', torch.zeros(num_features))

    def _update(self, x):
        self._count += 1
        self.mean.data += (x.mean(dim=0) - self.mean) / self._count
        self.variance.data += (((x - self.mean) ** 2).mean(dim=0) - self.variance) / self._count

    def forward(self, x):
        if self.training:
            self._update(x)
        if self._count > 1:
            return (x - self.mean) / (self.variance ** 0.5 + 1e-9)
        else:
            return x - self.mean
