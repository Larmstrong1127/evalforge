"""Post-hoc temperature calibration for Bradley-Terry reward margins.

Bradley-Terry logits are arbitrarily scaled: a confident model can emit
rewards like +/-8, which a bare sigmoid flattens to 0.999/0.001 and
destroys score granularity. We fit a single scalar T on the ID validation
split by minimizing the NLL of sigma(margin / T) — every margin in the
calibration set is a (chosen - rejected) pair, so the "label" is always 1
(chosen preferred) and miscalibration shows up when large margins coexist
with preference violations (negative margins).

Fitting T on the same split used for ID evaluation is safe for the
headline pairwise-accuracy metric: accuracy is invariant under any
positive temperature (monotone transform).
"""
import torch


def fit_temperature(margins: torch.Tensor, steps: int = 200, lr: float = 0.05) -> float:
    """Fit scalar T minimizing mean softplus(-margins / T) with Adam.

    Optimizes log_t for positivity. Margins: r_chosen - r_rejected on the
    calibration split.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.softplus(-margins / log_t.exp()).mean()
        loss.backward()
        optimizer.step()
    return float(log_t.exp().item())
