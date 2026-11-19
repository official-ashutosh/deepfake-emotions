"""
Gradient Reversal Layer (GRL) and lambda scheduling.
Based on Ganin et al., Domain-Adversarial Training of Neural Networks (JMLR 2016).
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = float(lambda_val)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Multiplies incoming gradient by -lambda
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = float(lambda_val)

    def set_lambda(self, lambda_val: float):
        self.lambda_val = float(lambda_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

    def __repr__(self):
        return f"GradientReversalLayer(lambda={self.lambda_val:.4f})"


def compute_dann_lambda(p: float, lambda_max: float = 1.0) -> float:
    """
    Standard DANN sigmoid schedule:
    lambda(p) = lambda_max * (2.0 / (1.0 + exp(-10 * p)) - 1.0)
    where p is training progress in [0, 1].
    """
    p = max(0.0, min(1.0, float(p)))
    gamma = 10.0
    val = (2.0 / (1.0 + math.exp(-gamma * p))) - 1.0
    return float(lambda_max * val)
