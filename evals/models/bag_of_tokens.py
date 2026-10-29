import torch
from torch import nn
from torch.nn import functional as F


class BagOfTokensClassifier(nn.Module):
    """Logistic regression on token counts divided by vocab_size.

    Input: torch.long token IDs in [0, vocab_size), shaped (batch_size, ...).
    Output: (batch_size, n_classes); binary logits are [0, z], with one learned score.
    loss() adds L2 regularization to weight; smaller C means a stronger penalty.
    """

    def __init__(self, vocab_size, C=1.0, n_classes=2):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not C > 0:
            raise ValueError("C must be positive")
        assert isinstance(n_classes, int) and n_classes >= 2
        self.vocab_size = vocab_size
        self.C = C
        self.n_classes = n_classes
        n_outputs = 1 if n_classes == 2 else n_classes
        self.weight = nn.Parameter(torch.zeros(vocab_size, n_outputs))
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, token_ids):
        token_ids = token_ids.flatten(start_dim=1)

        # Count each token ID across all positions in each example.
        counts = self.weight.new_zeros((token_ids.shape[0], self.vocab_size))
        counts.scatter_add_(
            1, token_ids, torch.ones_like(token_ids, dtype=counts.dtype)
        )
        features = counts / self.vocab_size
        logits = features @ self.weight + self.bias
        if self.n_classes == 2:
            logits = torch.cat((torch.zeros_like(logits), logits), dim=1)
        return logits

    def loss(self, logits, targets, n_train):
        """Mean cross-entropy + L2, with sklearn's unweighted C scaling.

        targets contains class labels in [0, n_classes).
        n_train is the full training-set size, even when using minibatches.
        Only weight is penalized; bias is unpenalized.
        """
        if n_train < 1:
            raise ValueError("n_train must be positive")
        cross_entropy = F.cross_entropy(logits, targets.long())
        l2 = self.weight.square().sum() / (2 * self.C * n_train)
        return cross_entropy + l2
