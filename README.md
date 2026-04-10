# lorenz-project-shmc5205

Lorenz63 model integration and ensemble analysis for ATOC 4815/5815.  
This package implements the Lorenz‑63 dynamical system, simple numerical integration tools, ensemble forecasting, and plotting utilities.  
It also exposes a command‑line tool, **`run-lorenz`**, which generates an ensemble predictability figure.

---

## Installation

```bash
pip install lorenz-project-shmc5205
```

Or from source:

```bash
git clone https://github.com/Shannon-Harper/lorenz_ens_SHM.git
cd lorenz_ens_SHM
pip install -e .
```

## Quick Start

```python
from lorenz_project import Lorenz63

model = Lorenz63(sigma=10, rho=28, beta=8/3)
trajectory = model.run([1, 1, 1], dt=0.01, n_steps=5000)
```

## Command Line

```bash
run-lorenz    # generates lorenz_ensemble_predictability.png

```

## License

MIT