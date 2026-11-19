"""
Unit tests for Gradient Reversal Layer (GRL).
"""

import math
import torch
import pytest
from models.grl import GradientReversalLayer, compute_dann_lambda


def test_grl_forward_is_identity():
    x = torch.randn(4, 64, requires_grad=True)
    grl = GradientReversalLayer(lambda_val=0.5)
    out = grl(x)
    assert torch.equal(out, x), "GRL forward must be exact identity"


def test_grl_backward_reverses_gradient():
    x = torch.randn(4, 64, requires_grad=True)
    lambda_val = 0.75
    grl = GradientReversalLayer(lambda_val=lambda_val)
    
    out = grl(x)
    loss = (out ** 2).sum()
    loss.backward()

    # Analytical gradient of (x^2).sum() w.r.t x is 2 * x
    # Through GRL, the gradient arriving at x should be -lambda * 2 * x
    expected_grad = -lambda_val * (2.0 * x.detach())
    diff = (x.grad - expected_grad).abs().max().item()
    assert diff < 1e-6, f"Backward gradient mismatch: max diff {diff}"


def test_grl_dynamic_lambda_update():
    x = torch.randn(2, 10, requires_grad=True)
    grl = GradientReversalLayer(lambda_val=1.0)
    
    grl.set_lambda(0.2)
    out = grl(x)
    loss = out.sum()
    loss.backward()
    
    # expected grad = -0.2 * ones
    expected_grad = -0.2 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


def test_dann_lambda_schedule():
    l_max = 1.0
    val_0 = compute_dann_lambda(0.0, lambda_max=l_max)
    assert abs(val_0 - 0.0) < 1e-7, "At p=0, lambda should be 0"

    val_mid = compute_dann_lambda(0.5, lambda_max=l_max)
    assert 0.0 < val_mid < l_max, "At p=0.5, lambda should be intermediate"

    val_1 = compute_dann_lambda(1.0, lambda_max=l_max)
    # 2 / (1 + exp(-10)) - 1 = 2 / (1 + 4.54e-5) - 1 ~ 0.9999
    assert abs(val_1 - l_max) < 0.01, f"At p=1.0, lambda should approach {l_max}"

    # Monotonicity test
    steps = [compute_dann_lambda(p / 10.0, lambda_max=0.5) for p in range(11)]
    for i in range(len(steps) - 1):
        assert steps[i] <= steps[i + 1], "Schedule must be monotonically increasing"


if __name__ == "__main__":
    test_grl_forward_is_identity()
    test_grl_backward_reverses_gradient()
    test_grl_dynamic_lambda_update()
    test_dann_lambda_schedule()
    print("All GRL unit tests passed successfully!")
