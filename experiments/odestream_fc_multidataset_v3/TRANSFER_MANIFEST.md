# v3 cluster transfer and dependency audit

Audit date: 2026-10-01. No scientific computation, installation, Git mutation,
dataset modification, historical modification, or pilot-output deletion was
performed. All paths below are repository-relative.

## Git status limitation

`git status --short` and `git ls-files` both fail with "not a git repository" in
the supplied workspace. Every tracking status below is **UNKNOWN**, not an
assertion that a file is untracked. A/B required tracked/untracked source files,
root/dataset tracking, unrelated modifications, and accidentally tracked runtime
artifacts cannot be classified until an actual checkout is available. This is a
blocker to declaring Git transfer ready. No Git repository was initialized.

## Exact must-push set

All following files must be present in the reviewed repository commit. Existing
files need no new commit if already identical there. Tracking status for every
listed path: UNKNOWN due to the missing Git metadata.

### A. Existing v3 source and documentation

Each filename in this table has the exact prefix
`experiments/odestream_fc_multidataset_v3/`.

| File | Why required |
|---|---|
| `__init__.py` | v3 package |
| `run.py` | CLI and frozen-plan selection |
| `config.py` | protocol, dataset registry, master DAG |
| `datasets.py` | exact data loading/scaling/window semantics |
| `runner.py` | orchestration, locking, artifact validation/resume |
| `engine.py` | warmup/bootstrap/online scientific loops |
| `runtime.py` | CPU threads, RNG, numerical helpers |
| `model_adapter.py` | model construction and boundary transitions |
| `models.py` | extracted VAE/LSTM/fusion architecture |
| `neural_ode.py` | extracted custom Neural ODE/adjoint |
| `losses.py` | PaperLoss-B, optimizer, finite checks |
| `fc.py` | student/teacher, same noise, EMA, FC update |
| `r50.py` | rolling median gate |
| `calibration.py` | reference and rho-to-lambda derivation |
| `checkpointing.py` | atomic persistence, hashes, timestamps |
| `provenance.py` | data/source integrity, environment metadata |
| `timing.py` | scientific timer boundary |
| `reporting.py` | scientific aggregation and report output |
| `status.py` | read-only progress projection |
| `static_validate.py` | static review utility; also included in source manifest |
| `README.md` | canonical usage documentation |
| `STATIC_REVIEW.md` | implementation review evidence |

### B. Existing root source dependencies

**None.** `experiments/` is a namespace package; it has no `__init__.py` and does
not need one when commands are run from the repository root. v3's own
`__init__.py` must be transferred. No root `ODEVAE.py`, `NeuralODE.py`, `data.py`,
`TIL.py`, model directory, root requirements, or historical experiment code is
imported transitively. No historical outputs/checkpoints/readiness artifacts
are opened by the runtime. ECL's provenance description mentions its original
conversion source, but the runtime reads only `data/ECL.csv`.

### C. Exact dataset files

| Path | Bytes | SHA256 | Tracking | GitHub storage |
|---|---:|---|---|---|
| `data/ETTh2.csv` | 2417960 | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` | UNKNOWN | Ordinary Git suitable |
| `data/ECL.csv` | 95557815 | `583f9538d489bfd29ba5edb3f3c2b51152ad8900468a8bbc17e1f0a6f2118ffd` | UNKNOWN | 91.13 MiB; ordinary Git permitted, size warning |
| `data/Weather.csv` | 7235425 | `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63` | UNKNOWN | Ordinary Git suitable |

All three hashes were recomputed and match `config.registry()`. No rows or bytes
were changed. GitHub warns above 50 MiB and blocks ordinary files above 100 MiB:
[official size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
None of these files requires LFS by size. ECL could optionally use LFS, but no
LFS policy/tracking was introduced. If the actual repository already uses LFS,
install Git LFS and fetch real file content before the hash check. Avoid text
conversion/filter rules that change dataset bytes. Do not regenerate ECL or
normalize Weather. `electricity.txt.gz` is not a v3 runtime dependency.

### D. New/modified packaging paths

| Path | Change | Tracking |
|---|---|---|
| `.gitignore` | Added `.idea/`, `.DS_Store`, `/outputs/` | UNKNOWN |
| `experiments/odestream_fc_multidataset_v3/requirements.txt` | New pinned Linux dependency closure | UNKNOWN |
| `experiments/odestream_fc_multidataset_v3/cluster_preflight.sh` | New read-only deployment checker | UNKNOWN |
| `experiments/odestream_fc_multidataset_v3/CLUSTER_SETUP.md` | New server workflow | UNKNOWN |
| `experiments/odestream_fc_multidataset_v3/MAC_PILOT_ENVIRONMENT.md` | New sanitized provenance | UNKNOWN |
| `experiments/odestream_fc_multidataset_v3/TRANSFER_MANIFEST.md` | This audit and transfer list | UNKNOWN |

The new ignore patterns cannot match any source or dataset in A/C/D. Existing
rules already exclude `.venv`, `venv/`, `__pycache__/`, `*.py[cod]`, `.cache`,
`.pytest_cache/`, `.mypy_cache/`, `*.log`, and common build/temp environments.
v3 checkpoint/trace/temp/progress files all fall under `/outputs/`. No global
`*.pt` rule was added, avoiding accidental hiding of historical evidence.
Actual ignored-versus-tracked status still requires Git review: ignore rules
do not exclude files already tracked.

## Transitive import and requirements self-check

Entry graph: `run -> config, checkpointing`; explicit `run` additionally imports
`runner -> provenance, runtime, datasets, engine, calibration, reporting`.
`engine/calibration -> model_adapter, losses, fc, r50, timing`;
`model_adapter/fc -> models -> neural_ode`. `status -> config, checkpointing`.
All repository imports stay inside v3, including lazy function-local imports.

### A. Scientific execution and its dependency closure

| Distribution | Installed Mac / Linux pin | Import evidence / dependency parent | Linux CPython 3.10 assessment |
|---|---|---|---|
| torch | 1.12.0 / 1.12.0+cpu | models, neural_ode, fc, engine, losses, runtime, checkpointing | Official cp310 Linux x86_64 CPU wheel; explicit platform build change |
| numpy | 1.23.5 / same | datasets, neural_ode, losses, engine, calibration, r50, runtime, reporting | cp310 manylinux x86_64 wheel |
| pandas | 1.5.3 / same | datasets.load_data; provenance | cp310 manylinux x86_64 wheel |
| scikit-learn | 1.2.2 / same | datasets.load_data imports sklearn.preprocessing.StandardScaler | cp310 manylinux x86_64 wheel |
| scipy | 1.10.1 / same | required by sklearn import/package metadata | cp310 manylinux wheel; NumPy 1.23.5 satisfies >=1.19.5,<1.27 |
| joblib | 1.6.0 / same | sklearn required dependency | universal wheel; Python >=3.10 |
| cloudpickle | 3.1.2 / same | joblib 1.6.0 requires >=3.0 | universal wheel; Python >=3.8 |
| threadpoolctl | 3.7.0 / same | sklearn required dependency | universal wheel; Python >=3.9 |
| python-dateutil | 2.9.0.post0 / same | pandas required dependency | universal wheel |
| pytz | 2026.3.post1 / same | pandas required dependency | universal wheel |
| six | 1.17.0 / same | python-dateutil required dependency | universal wheel |
| typing_extensions | 4.16.0 / same | torch required dependency | universal wheel; Python >=3.9 |

These are the actual required dependency closure, not a copy of `pip freeze`.
No extras such as package test/docs dependencies were included. Binary-only
installation avoids silently building a different local scientific stack.

### B. Reporting/status and provenance

Reporting reuses NumPy; status and plan need only the standard library.
`psutil==5.9.8` is included solely for optional physical-core provenance in
`provenance.environment`. It matches the current Mac environment and has a
Linux x86_64 abi3 wheel compatible with CPython 3.10. It is not needed for model
mathematics, but including it preserves the requested machine metadata.
There are no plotting dependencies.

### C. Standard library (never added to requirements)

argparse, ast, collections, contextlib, copy, csv, datetime, fcntl, gzip, hashlib,
io, json, math, os, pathlib, platform, random, signal, socket, subprocess, sys,
tempfile, time. `fcntl` is available on Linux/macOS; Windows is not supported.
The shell preflight additionally uses standard-library importlib.metadata.

### D. Historical/unused exclusions

torchdiffeq (not installed; v3 implements its ODE internally), torchvision,
torchaudio, matplotlib, seaborn, IPython/Jupyter, tqdm, memory-profiler, Optuna,
IDE tooling, and Mac-specific packages are unnecessary. Root reproduction pins
are a separate environment specification, not an input to v3 installation.

Compatibility evidence: official [PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/),
[NumPy](https://pypi.org/project/numpy/1.23.5/#files),
[pandas](https://pypi.org/project/pandas/1.5.3/#files),
[scikit-learn](https://pypi.org/project/scikit-learn/1.2.2/#files),
[SciPy](https://pypi.org/project/scipy/1.10.1/#files),
[psutil](https://pypi.org/project/psutil/5.9.8/#files),
[joblib](https://pypi.org/project/joblib/1.6.0/),
[cloudpickle](https://pypi.org/project/cloudpickle/3.1.2/),
[threadpoolctl](https://pypi.org/project/threadpoolctl/3.7.0/),
[dateutil](https://pypi.org/project/python-dateutil/2.9.0.post0/),
[pytz](https://pypi.org/project/pytz/2026.3.post1/),
[six](https://pypi.org/project/six/1.17.0/),
[typing_extensions](https://pypi.org/project/typing-extensions/4.16.0/).
Availability and metadata were inspected; no Linux installation was executed.
Target architecture, glibc and successful imports must be confirmed on paranazao.

## Exact must-not-push categories

- `.venv/`, `venv/`, other local environments: platform-specific executables and packages.
- `__pycache__/`, `*.pyc`, `.cache/`, `.pytest_cache/`, `.mypy_cache/`, build/wheel caches: generated machine state.
- `.idea/`, `.DS_Store`: IDE/OS metadata.
- `outputs/` in its entirety: validation-only and future machine-local scientific state.
- Local execution `logs/`, `*.log`, temporary files, checkpoints, generated results:
  not deployment inputs. Do not broadly add existing `runs/`, `results/`,
  `diagnostics/`, or benchmark archives; their tracking/history was not auditable.
- Historical experiment artifacts: reference evidence, no new changes for this
  package and no runtime dependency. Preserve existing repository history.

### Mac pilot inventory: LOCAL VALIDATION ONLY — DO NOT PUSH

All paths in the following inventory are under
`outputs/odestream-fc-sensitivity-v3/`; exclude the entire directory, not just
the two results. The state is PENDING / SUBSET_COMPLETE, with 2/90 scientific
warmups and 2/90 online trajectories complete. Initial cluster state must be 0/90.

```text
protocol.json
plan.json
sources.json
state.json
execution_environment.json
execution_sessions/2026-10-01T04-46-38.347584+00-00.json
runner.lock
datasets/ETTh2/dataset_manifest.json
datasets/Weather/dataset_manifest.json
datasets/ECL/dataset_manifest.json
datasets/ECL/preparation/control_seed_0/state.pt
datasets/ECL/preparation/control_seed_0.lineage.json
datasets/ECL/calibration/reference.json
datasets/ECL/calibration/reference.state.pt
datasets/ECL/calibration/reference.lineage.json
datasets/ECL/calibration/rho_1.0.json
datasets/ECL/calibration/rho_1.0.lineage.json
datasets/ECL/control/seed_0/warmup/state.pt
datasets/ECL/control/seed_0/warmup.lineage.json
datasets/ECL/control/seed_0/always/state.pt
datasets/ECL/control/seed_0/always/result.json
datasets/ECL/control/seed_0/always/trace.csv.gz
datasets/ECL/control/seed_0/always.lineage.json
datasets/ECL/fc/rho_1.0/seed_0/warmup/state.pt
datasets/ECL/fc/rho_1.0/seed_0/warmup.lineage.json
datasets/ECL/fc/rho_1.0/seed_0/bootstrap/state.pt
datasets/ECL/fc/rho_1.0/seed_0/bootstrap.lineage.json
datasets/ECL/fc/rho_1.0/seed_0/r50/state.pt
datasets/ECL/fc/rho_1.0/seed_0/r50/result.json
datasets/ECL/fc/rho_1.0/seed_0/r50/trace.csv.gz
datasets/ECL/fc/rho_1.0/seed_0/r50.lineage.json
```

No `reports/` files or completion manifest were present in this output inventory;
if later generated they also remain excluded. Any console logs outside this
directory are likewise local validation only. No local artifacts were removed.

## Clean-clone conclusion

The listed source/data/dependency set is self-contained and uses repository-
relative paths. Existing Python science, source identity, datasets and pilot
artifacts were left intact. The 90-job plan is not changed by packaging.
Reliable Git transfer remains BLOCKED until actual Git tracking and the reviewed
commit are available. The specified Linux x86_64/Python 3.10.6 platform must also
pass the supplied non-scientific checks before any authorized scientific run.

Static validation completed locally: shell syntax passed; `--local-audit`
passed all dependency imports/version checks, dataset SHA256 checks, CPU 8/8
configuration and plan counts. Installed distribution metadata confirms that
every non-extra dependency is included and its version constraint is satisfied.
The full plan retained 90 trajectories and the ECL seed0/rho1 selection contained
8 dependency nodes / 2 online jobs. Plan/status imported no PyTorch and left
pilot file sizes/mtime values unchanged. All 20 original v3 Python SHA256 values
are unchanged. No scientific `run` was invoked. Historical folders were never
written. Linux installation and Git tracking verification remain unperformed.
