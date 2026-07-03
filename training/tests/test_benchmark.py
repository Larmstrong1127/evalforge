import pytest

from training.benchmark import BenchmarkResult, aggregate_benchmark_results


def test_aggregate_computes_agreement_cost_and_latency():
    # Ground truth labels: [0, 1, 1]
    # Local judge predictions: [0, 1, 0] -> 2/3 correct
    # Claude judge predictions: [0, 1, 1] -> 3/3 correct
    ground_truth = [0, 1, 1]
    local_preds = [0, 1, 0]
    local_costs = [0.0, 0.0, 0.0]
    local_latencies_ms = [45.0, 50.0, 48.0]

    claude_preds = [0, 1, 1]
    claude_costs = [0.002, 0.002, 0.002]
    claude_latencies_ms = [800.0, 820.0, 810.0]

    result = aggregate_benchmark_results(
        ground_truth=ground_truth,
        judges={
            "local-deberta": BenchmarkResult(
                predictions=local_preds, costs_usd=local_costs, latencies_ms=local_latencies_ms
            ),
            "claude-sonnet-5": BenchmarkResult(
                predictions=claude_preds, costs_usd=claude_costs, latencies_ms=claude_latencies_ms
            ),
        },
    )

    assert result["local-deberta"]["agreement"] == pytest.approx(2 / 3)
    assert result["claude-sonnet-5"]["agreement"] == pytest.approx(1.0)
    assert result["local-deberta"]["total_cost_usd"] == pytest.approx(0.0)
    assert result["claude-sonnet-5"]["total_cost_usd"] == pytest.approx(0.006)
    assert result["local-deberta"]["p50_latency_ms"] == pytest.approx(48.0)


def test_aggregate_computes_cost_per_1k():
    result = aggregate_benchmark_results(
        ground_truth=[0, 1],
        judges={
            "claude-sonnet-5": BenchmarkResult(
                predictions=[0, 1], costs_usd=[0.002, 0.002], latencies_ms=[800.0, 810.0]
            ),
        },
    )
    # avg cost per item = 0.002, * 1000 = 2.0
    assert result["claude-sonnet-5"]["cost_per_1k_usd"] == pytest.approx(2.0)


def test_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        aggregate_benchmark_results(
            ground_truth=[0, 1, 1],
            judges={
                "bad-judge": BenchmarkResult(
                    predictions=[0, 1], costs_usd=[0.0, 0.0], latencies_ms=[1.0, 1.0]
                ),
            },
        )
