import numpy as np
import pytest
from causal_comparator.utils import SHD_vectorized


def test_shd_vectorized_two_edge_difference():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((3, 3))

    # true: 0 -> 1 -> 2
    B_true[1, 0] = 1
    B_true[2, 1] = 1

    # predicted: 2 <- 0 -> 1
    B_predicted[2, 0] = 1
    B_predicted[1, 0] = 1

    assert SHD_vectorized(B_true, B_predicted) == 2


def test_shd_vectorized_reversal_double_penalty():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((3, 3))

    # true: 0 -> 1 -> 2
    B_true[1, 0] = 1
    B_true[2, 1] = 1

    # predicted: 0 <- 1 -> 2
    B_predicted[0, 1] = 1
    B_predicted[2, 1] = 1

    assert SHD_vectorized(B_true, B_predicted, double_for_anticausal=True) == 2


def test_shd_vectorized_identical_graphs():
    B_true = np.zeros((3, 3))
    B_true[1, 0] = 1
    B_true[2, 1] = 1

    B_predicted = B_true.copy()

    assert SHD_vectorized(B_true, B_predicted) == 0


def test_shd_vectorized_one_missing_edge():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((3, 3))

    B_true[1, 0] = 1
    B_true[2, 1] = 1

    B_predicted[1, 0] = 1

    assert SHD_vectorized(B_true, B_predicted) == 1


def test_shd_vectorized_one_extra_edge():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((3, 3))

    B_true[1, 0] = 1

    B_predicted[1, 0] = 1
    B_predicted[2, 0] = 1

    assert SHD_vectorized(B_true, B_predicted) == 1


def test_shd_vectorized_reversal_single_penalty():
    B_true = np.zeros((2, 2))
    B_predicted = np.zeros((2, 2))

    # true: 0 -> 1
    B_true[1, 0] = 1

    # predicted: 1 -> 0
    B_predicted[0, 1] = 1

    assert SHD_vectorized(B_true, B_predicted, double_for_anticausal=False) == 1


def test_shd_vectorized_reversal_double_penalty_simple():
    B_true = np.zeros((2, 2))
    B_predicted = np.zeros((2, 2))

    B_true[1, 0] = 1
    B_predicted[0, 1] = 1

    assert SHD_vectorized(B_true, B_predicted, double_for_anticausal=True) == 2


def test_shd_vectorized_empty_graphs():
    B_true = np.zeros((4, 4))
    B_predicted = np.zeros((4, 4))

    assert SHD_vectorized(B_true, B_predicted) == 0


def test_shd_vectorized_threshold_behavior():
    B_true = np.zeros((2, 2))
    B_predicted = np.zeros((2, 2))

    B_true[1, 0] = 1e-9

    assert SHD_vectorized(B_true, B_predicted, threshold=1e-8) == 0
    assert SHD_vectorized(B_true, B_predicted, threshold=1e-10) == 1


def test_shd_vectorized_shape_mismatch():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((4, 4))

    with pytest.raises(ValueError):
        SHD_vectorized(B_true, B_predicted)


def test_shd_vectorized_symmetry():
    B_true = np.zeros((3, 3))
    B_predicted = np.zeros((3, 3))

    B_true[1, 0] = 1
    B_true[2, 1] = 1

    B_predicted[1, 0] = 1
    B_predicted[2, 0] = 1

    s1 = SHD_vectorized(B_true, B_predicted)
    s2 = SHD_vectorized(B_predicted, B_true)

    assert s1 == s2

@pytest.mark.parametrize("seed", range(50))
def test_shd_vectorized_matches_reference_like(seed):
    """
    Stress test on DAG-like matrices (LiNGAM setting)
    """

    rng = np.random.default_rng(seed)

    B_true = np.tril((rng.random((10, 10)) < 0.5).astype(int), k=-1)
    B_predicted = np.tril((rng.random((10, 10)) < 0.5).astype(int), k=-1)

    for flag in [True, False]:
        s1 = SHD_vectorized(B_true, B_predicted, double_for_anticausal=flag)
        s2 = SHD_vectorized(B_predicted, B_true, double_for_anticausal=flag)

        # basic sanity: symmetry
        assert s1 == s2, f"SHD1:{s1} is not equal to SHD2: {s2} with seed: {seed}, flag: {flag}"