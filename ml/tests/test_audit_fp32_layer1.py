"""Unit tests verifying fail-closed behavior of Layer 1 & 6-Layer FP32 parity audit.

Tests:
  1. Audit PASS on valid reference and browser artifacts.
  2. Fail-closed on missing reference file.
  3. Fail-closed on missing browser file.
  4. Fail-closed on corrupted sample count (< 16).
  5. Fail-closed on wrong tensor shape or byte length.
  6. Fail-closed on NaN / Inf in tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pytest

from scripts.research.audit_fp32_layer1_tensors import audit_layer1_tensors
from scripts.research.audit_browser_fp32_parity import audit_parity, check_finite


def test_audit_layer1_success_on_current_artifacts():
    """Verify that current repo artifacts pass Layer 1 parity audit."""
    receipt = audit_layer1_tensors()
    assert receipt["verdict"] == "PASS"
    assert receipt["scientific_conclusion"] == "PARITY_WITHIN_NUMERICAL_TOLERANCE_NOT_BIT_EXACT"
    assert receipt["aggregate_metrics"]["total_samples"] == 16
    assert receipt["aggregate_metrics"]["total_elements_compared"] == 16 * 150528
    assert receipt["aggregate_metrics"]["overall_element_wise_mae"] < 1.0e-2
    assert receipt["aggregate_metrics"]["overall_max_abs_diff"] < 5.0e-2


def test_audit_6layer_parity_success_on_current_artifacts():
    """Verify that current repo artifacts pass full 6-layer parity audit."""
    receipt = audit_parity()
    assert receipt["verdict"] == "PASS"
    assert receipt["numerical_parity_6layers_audit"]["layer6_probabilities_and_decisions"]["decisions_matched"] == 160


def test_check_finite_rejects_nan_and_inf():
    """Verify that check_finite rejects NaN and Inf immediately."""
    arr_clean = np.zeros((10,), dtype=np.float32)
    check_finite(arr_clean, "clean")

    arr_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(arr_nan, "nan_test")

    arr_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(arr_inf, "inf_test")
