from nes.norm_repository import NormRepository

def test_norm_repo_store_and_retrieve():
    repo = NormRepository()
    repo.add_norm("honesty", weight=1.0)
    entry = repo.get_norm("honesty")
    assert entry["weight"] == 1.0

def test_norm_veto_tag():
    repo = NormRepository()
    repo.add_norm("stop", weight=0.5, veto=True)
    entry = repo.get_norm("stop")
    assert entry["veto"] is True
