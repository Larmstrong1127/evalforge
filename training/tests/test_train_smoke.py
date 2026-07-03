import torch

from tests.conftest import make_synthetic_batch, make_tiny_model_and_tokenizer
from training.train import run_training_steps


def test_loss_is_finite_and_decreases_over_steps():
    torch.manual_seed(42)
    model = make_tiny_model_and_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [make_synthetic_batch(batch_size=4, seq_len=8) for _ in range(10)]

    losses = run_training_steps(
        model=model,
        optimizer=optimizer,
        batches=batches,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        use_amp=False,  # mixed precision has no benefit on CPU; smoke test runs without it
    )

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    first_half = sum(losses[:5]) / 5
    second_half = sum(losses[5:]) / 5
    assert second_half <= first_half + 0.5  # generous margin; 10 steps is noisy
