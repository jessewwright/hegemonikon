from nes.assent_gate import AssentGate

def test_assent_gate_thresholding():
    gate = AssentGate(base_threshold=1.0)
    assert gate.evaluate(impulse_strength=0.5) is False
    assert gate.evaluate(impulse_strength=1.5) is True
