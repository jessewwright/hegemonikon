# Hegemonikon

[![CI](https://github.com/jessewwright/hegemonikon/actions/workflows/ci.yml/badge.svg)](https://github.com/jessewwright/hegemonikon/actions)

## Getting Started

```bash
git clone https://github.com/jessewwright/hegemonikon.git
cd hegemonikon

# Create a venv & install
python -m venv .venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Run a quick Stroop sim
nes-run stroop --params params/stroop_default.json --n-trials 100
```

## Examples

- **Stroop demo**: [`notebooks/Stroop_demo.ipynb`](notebooks/Stroop_demo.ipynb)
