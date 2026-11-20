import numpy as np

from src.calibrate import ece, fit_temperature, nll
from src.evaluate import sigmoid


def _overconfident_logits(n=20000, seed=0):
    """Labels plus logits whose true log-odds are z_true; the model reports 3*z_true."""
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0, 2, n)
    y = (rng.random(n) < sigmoid(z_true)).astype(int)
    return 3 * z_true, y


def test_fit_temperature_recovers_overconfidence():
    z, y = _overconfident_logits()
    assert abs(fit_temperature(z, y) - 3.0) < 0.3


def test_temperature_lowers_nll_and_ece_but_not_accuracy():
    z, y = _overconfident_logits()
    T = fit_temperature(z, y)
    assert nll(z, y, T) < nll(z, y, 1.0)
    assert ece(sigmoid(z / T), y) < ece(sigmoid(z), y)
    # dividing by T > 0 never flips a prediction
    assert np.array_equal(sigmoid(z) >= 0.5, sigmoid(z / T) >= 0.5)


def test_ece_zero_when_perfect():
    y = np.array([0, 1, 0, 1])
    assert ece(np.array([0.0, 1.0, 0.0, 1.0]), y) == 0.0


def test_nll_finite_for_extreme_logits():
    z = np.array([60.0, -60.0, 60.0], dtype=np.float32)  # sigmoid rounds to exactly 0/1 in float32
    y = np.array([1, 0, 0])
    v = nll(z, y, 1.0)
    assert np.isfinite(v) and abs(v - 20.0) < 1e-6  # one confident mistake costs 60, mean = 20
