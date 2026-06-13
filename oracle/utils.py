import torch
import torch.nn as nn
import numpy as np

def compute_measurement_stds(dataset, num_measurements):
    """Compute standard deviation of delta_y across the training set."""
    all_deltas = []
    for i in range(len(dataset)):
        _, delta = dataset[i]
        all_deltas.append(delta.numpy())
    all_deltas = np.array(all_deltas)  # (N, 14)
    stds = np.std(all_deltas, axis=0) + 1e-8
    return torch.FloatTensor(stds)

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, stds, beta=1.0):
        super().__init__()
        self.stds = stds
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, pred, target):
        # pred, target: (B, num_measurements)
        loss_per_element = self.loss_fn(pred, target)   # (B, num_measurements)
        # weight by inverse std
        weights = 1.0 / self.stds.to(pred.device)
        weighted_loss = loss_per_element * weights
        return weighted_loss.mean()
