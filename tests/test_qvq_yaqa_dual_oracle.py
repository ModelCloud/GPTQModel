import torch

from scripts.validate_qvq_yaqa_p32_dual_oracle import _gate_delta, _objective


def test_qvq_yaqa_dual_oracle_matches_direct_trace_in_both_precisions():
    reconstructed = torch.tensor([[1.25, -0.5], [0.75, 2.0]])
    target = torch.tensor([[1.0, -0.25], [0.5, 1.5]])
    h_input = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    h_output = torch.tensor([[1.5, -0.125], [-0.125, 0.75]])

    for dtype in (torch.float32, torch.float64):
        error = reconstructed.to(dtype) - target.to(dtype)
        expected = torch.trace(error.transpose(0, 1) @ h_input.to(dtype) @ error @ h_output.to(dtype))
        actual = _objective(reconstructed, target, h_input, h_output, dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qvq_yaqa_dual_oracle_accepts_only_configured_regression_budget():
    exact = _gate_delta(10.0, 10.0, absolute=0.0, relative=0.0)
    bounded = _gate_delta(10.0, 10.05, absolute=0.0, relative=0.01)
    rejected = _gate_delta(10.0, 10.11, absolute=0.1, relative=0.01)

    assert exact["passed"] is True
    assert bounded["passed"] is True
    assert rejected["passed"] is False
