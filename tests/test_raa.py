from nes.raa import RAA

def test_raa_initial_weights():
    # RAA should initialize with default or provided weight vector
    raa = RAA(weights=[0.2, 0.8])
    assert hasattr(raa, "weights")
    assert len(raa.weights) == 2

def test_raa_update():
    raa = RAA(weights=[0.5])
    before = raa.weights[0]
    raa.update(0, reward=1.0, alpha=0.1)
    after = raa.weights[0]
    assert after != before
