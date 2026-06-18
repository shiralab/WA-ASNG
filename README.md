# WA-ASNG: Weight Adaptation Adaptive Stochastic Natural Gradient

This repository contains the reference implementation and experiment code for
**WA-ASNG (Weight Adaptation Adaptive Stochastic Natural Gradient)**.
WA-ASNG extends ASNG (Adaptive Stochastic Natural Gradient) by adapting the weights used in the natural-gradient estimate.
The code implementation is based on CatCMA, provided in the Python library [cmaes](https://github.com/CyberAgentAILab/cmaes) by CyberAgent AI Lab.

---

## Repository structure

```
.
├── main.py
├── example/
│   ├── setting.json            # Example configuration (noiseless)
│   └── setting_noisy.json      # Example configuration (with noise)
├── src/
│   ├── algorithms/
│   │   ├── asng.py             # ASNG  (baseline)
│   │   ├── pbil.py             # PBIL  (baseline)
│   │   └── waasng.py           # WA-ASNG (proposed method)
│   ├── problems/
│   │   └── bench.py            # Benchmark functions (+ noisy variants)
│   └── utils/
│       └── logger.py           # CSV / config logger
├── pyproject.toml              # Project metadata and pinned dependencies
├── uv.lock                     # Locked dependency versions (for uv)
└── .python-version             # Target Python version (3.9.11)
```

Results are written to the `results/` directory at run time.

---

## Requirements

- **Python 3.9.11** (see [.python-version](.python-version))
- Dependencies are pinned in [pyproject.toml](pyproject.toml) and locked in [uv.lock](uv.lock)

> WA-ASNG uses [PyTorch](https://pytorch.org/) for the gradient-based weight adaptation. The pinned `torch` build targets Linux with CUDA libraries; on other platforms, you may need to install a platform-appropriate `torch` wheel. The code itself runs on CPU and sets `torch.set_num_threads(1)`.

---

## Environment setup

### uv (recommended)

This project is managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if needed (see https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install the locked dependencies
uv sync
```

### venv + pip

```bash
# Use Python 3.9.11 (e.g. via pyenv)
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# Install the project and its dependencies
pip install .
```

---

## Running an experiment

A single optimization run is launched by passing a JSON configuration file to
`main.py`:

```bash
# With uv
uv run python main.py example/setting.json

# Or, inside an activated virtual environment
python main.py example/setting.json
```

### Configuration file format

A configuration file has two sections, `problem` and `algorithm`:

```jsonc
{
    "problem": {
        "max_evals": 10000,        // evaluation budget
        "dim": 100,                // number of categorical variables
        "name": "NoisyOneMax",     // benchmark problem name (see below)
        "seed": 1,                 // random seed
        "noisevar": 1.0            // (noisy problems only) noise variance
    },
    "algorithm": {
        "name": "WAASNG",          // algorithm name (WAASNG | ASNG | PBIL)
        "param": {                 // keyword arguments forwarded to the algorithm
            "population_size": 100
        }
    }
}
```

Notes:

- If `name` starts with `Noisy`, the Gaussian noise is added to the objective function. The `noisevar` controls the Gaussian noise variance (default `0.0`).
- `algorithm.param` is passed directly to the chosen algorithm's constructor, so any of its keyword arguments can be set here (see the source files for the full list).

### Implemented algorithms

| Name     | Source                                  | Role             |
|----------|-----------------------------------------|------------------|
| `WAASNG` | [src/algorithms/waasng.py](src/algorithms/waasng.py) | Proposed method  |
| `ASNG`   | [src/algorithms/asng.py](src/algorithms/asng.py)     | Baseline         |
| `PBIL`   | [src/algorithms/pbil.py](src/algorithms/pbil.py)     | Baseline         |

### Implemented benchmark problems

Defined in [src/problems/catbench.py](src/problems/catbench.py):

| Name (config)        | Description                                  |
|----------------------|----------------------------------------------|
| `OneMax`             | OneMax                                        |
| `LeadingOnes`        | LeadingOnes                                   |
| `BinVal`             | Binary-value problem      |
| `NoisyOneMax`        | `OneMax` with additive noise      |
| `NoisyLeadingOnes`   | `LeadingOnes` with additive noise |
| `NoisyBinVal`        | `BinVal` with observation noise      |

---

## Output

Each run creates a directory under `results/` whose path encodes the algorithm,
its parameters, the problem, the dimension, and the seed:

```
results/<algorithm>/<param_string>/<problem>/dim_<dim>/seed_<seed>/
├── config.json      # The full configuration plus execution metadata 
├── execution.log    # Run-time log messages
└── log.csv          # Per-generation metrics
```

The columns logged in `log.csv` are managed in
[src/utils/logger.py](src/utils/logger.py).

---
