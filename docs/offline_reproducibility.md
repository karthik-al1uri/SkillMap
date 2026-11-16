# Offline verification and source export

`requirements.txt` declares direct runtime/notebook lower bounds. Plotly **5.19+** is required for `marker.cornerradius`; scikit-learn **1.5+** is required for `TSNE(max_iter=...)`. `threadpoolctl` is a direct dependency of bounded mining/evaluation scripts. `requirements-test.txt` includes runtime requirements plus pytest and packaging (used to inspect installed version constraints). Python **3.12** is the declared CI target/minimum; the present installed interpreter may be newer. These are compatibility constraints, not a lockfile or a tested minimum-version matrix.

No dependency installation is needed or authorized for checking an existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /absolute/path/to/existing/python -B -m scripts.check_offline_environment \
  --root /absolute/path/to/source --out-dir /private/tmp/new-skillmap-env-check --pytest
```

Run from the source root. The fresh output directory must be outside that root. It holds `environment.json`, matplotlib cache, and pytest temporary fixtures. The checker uses installed package metadata, then `pip check` without indexes/version checks. Tests run with Agg, numerical threads capped at one, pytest cache disabled, and an inherited Python socket/DNS guard. The guard chains the existing interpreter startup hook so system-site-package paths are preserved. A guard self-test intentionally attempts a localhost connection and verifies that Python raises before connecting. This guard is not an operating-system network sandbox; it does not constrain non-Python programs.

The checker does not train models or invoke pipeline entry points. The optional test suite includes small synthetic estimator fits where required by regression tests; it does not rebuild canonical artifacts or fit production models. No downloaded datasets or API credentials are needed for the focused suite. The hosted CI workflow declaratively installs test requirements before its guarded offline test step; that installation is not performed by the checker.

## Export uncommitted source without artifacts or secrets

A Git clone would omit uncommitted changes. Export an allowlisted filesystem snapshot instead, using a new external destination. Example, from the source root:

```bash
python -B - <<'PY'
from pathlib import Path
import shutil
source = Path.cwd().resolve()
export = Path('/private/tmp/new-skillmap-source-export')
export.mkdir()  # fail rather than overwrite any existing snapshot
for name in ('src', 'scripts', 'tests', 'docs', 'notebooks', '.github'):
    if (source/name).is_dir():
        shutil.copytree(source/name, export/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.ipynb_checkpoints', '.env', '.env.*'))
for name in ('requirements.txt', 'requirements-test.txt', 'README.md', 'CLAUDE.md', '.gitignore'):
    if (source/name).is_file():
        shutil.copyfile(source/name, export/name)
print(export)
PY
```

This includes the current uncommitted source, tests, and documentation; it excludes `.git`, canonical `data/` and `outputs/`, environment files, virtualenvs, credentials, and local machine configuration. Notebook saved outputs remain embedded historical evidence, not newly computed results. Inspect the export inventory if additional source directories or fixture types are introduced; do not broaden this to copying the entire repository.

Run the previous checker command from the exported root using the **same existing interpreter** and a second fresh external output directory. Passing proves the exported source can execute its offline tests without canonical data or secrets using the installed environment. It does **not** prove a fresh clone can install dependencies offline, establish an independently resolved environment, or reproduce historical mining/model results. A clean installation from a prebuilt wheelhouse and Python3.12 CI execution remain separate validation tasks requiring authorization and appropriate artifacts.
