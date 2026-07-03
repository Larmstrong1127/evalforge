import pytest

from training.metrics import compute_classification_metrics, expected_calibration_error


def test_compute_classification_metrics_hand_verified():
    # y_true: [1,1,0,0], y_pred: [1,0,0,0]
    # tp=1 (idx0), fn=1 (idx1), tn=2 (idx2,3), fp=0
    # precision = tp/(tp+fp) = 1/1 = 1.0
    # recall = tp/(tp+fn) = 1/2 = 0.5
    # f1 = 2*P*R/(P+R) = 2*1.0*0.5/1.5 = 0.6667
    result = compute_classification_metrics(y_true=[1, 1, 0, 0], y_pred=[1, 0, 0, 0])
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.6667, abs=1e-3)
    assert result.confusion_matrix == [[2, 0], [1, 1]]  # [[tn, fp], [fn, tp]]


def test_compute_classification_metrics_perfect_predictions():
    result = compute_classification_metrics(y_true=[1, 0, 1, 0], y_pred=[1, 0, 1, 0])
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_expected_calibration_error_single_bin_hand_verified():
    # 4 samples, all confidence 0.8, 3 correct 1 incorrect -> accuracy=0.75
    # single bin (0.8): ECE = |0.8 - 0.75| * (4/4) = 0.05
    ece = expected_calibration_error(
        confidences=[0.8, 0.8, 0.8, 0.8],
        correct=[True, True, True, False],
        n_bins=10,
    )
    assert ece == pytest.approx(0.05, abs=1e-6)


def test_expected_calibration_error_perfect_calibration_is_zero():
    # confidence exactly matches empirical accuracy in each bin
    ece = expected_calibration_error(
        confidences=[1.0, 1.0, 1.0, 1.0],
        correct=[True, True, True, True],
        n_bins=10,
    )
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_expected_calibration_error_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        expected_calibration_error(confidences=[], correct=[])


def test_expected_calibration_error_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        expected_calibration_error(confidences=[1.5], correct=[True])


def test_expected_calibration_error_rejects_negative_confidence():
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        expected_calibration_error(confidences=[-0.1], correct=[True])


def test_expected_calibration_error_multi_bin_weighted_average():
    # bin [0.5,0.6): confidences [0.5,0.5], mean conf=0.5, both correct ->
    #   acc=1.0, gap=0.5, weight=2/4
    # bin [0.9,1.0) index 9 (last bin): confidences [0.9,0.9], mean conf=0.9,
    #   1/2 correct -> acc=0.5, gap=0.4, weight=2/4
    # ECE = 0.5*0.5 + 0.5*0.4 = 0.25 + 0.2 = 0.45
    ece = expected_calibration_error(
        confidences=[0.5, 0.5, 0.9, 0.9],
        correct=[True, True, True, False],
        n_bins=10,
    )
    assert ece == pytest.approx(0.45, abs=1e-6)


def test_compute_classification_metrics_all_negative_class():
    result = compute_classification_metrics(y_true=[0, 0, 0], y_pred=[0, 0, 0])
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
