import pytest
from nes.comparator import Comparator

def test_simple_trial_runs():
    comp = Comparator(drift_rate=0.2, threshold=1.0, noise_std=0.05)
    result = comp.run_trial()
    assert isinstance(result, dict)
    assert "rt" in result and "choice" in result
    assert result["rt"] > 0
