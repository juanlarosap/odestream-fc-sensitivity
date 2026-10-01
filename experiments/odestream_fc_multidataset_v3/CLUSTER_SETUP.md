# v3 Linux CPU setup

Deployment target: paranazao, Linux x86_64 with glibc, CPython **3.10.6**.
Confirm the server architecture and interpreter before installing. Other
architectures and Python versions require review; do not resolve incompatibility
by silently changing scientific package versions. The scientific contract is
CPU ONLY, deterministic PyTorch, 8 intra-op and 8 inter-op threads.

Packaging audit limitation: the supplied local workspace has no `.git` metadata.
Tracking, remote/branch, pre-existing modifications, and accidentally tracked
outputs could not be verified. Before publishing, use an actual Git checkout
and review the exact paths in `TRANSFER_MANIFEST.md`. Ignore rules do not untrack
files already in Git. Do not commit/push outputs or copy them to the server.

## Fresh installation, after manual Git review

For a first checkout use `git clone <repository-url>` and enter that directory.
For an existing checkout, pull the reviewed commit. Commands below are for later
use on the server; no scientific run is part of setup.

```bash
cd <repository-root>
git pull --ff-only
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short
uname -m                       # expected x86_64
getconf GNU_LIBC_VERSION
python3.10 --version           # expected Python 3.10.6
test ! -e .venv                # stop if an environment already exists; do not overwrite it
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/odestream_fc_multidataset_v3/requirements.txt
python -m pip check
bash experiments/odestream_fc_multidataset_v3/cluster_preflight.sh
python -B -m experiments.odestream_fc_multidataset_v3.run plan
python -B -m experiments.odestream_fc_multidataset_v3.status
```

Use the administrator-provided Python 3.10.6 executable if it has a different
name. Do not substitute an arbitrary `python3`. Scientific code requires at
least Python 3.9 APIs (`Path.is_relative_to`, `str.removeprefix`, builtin generic
annotations); all v3 files parse with Python 3.9 grammar. The pinned dependency
closure requires at least 3.10 (joblib 1.6.0). Reproduce 3.10.6 for this study.
PyTorch 1.12.0 has a CPython 3.10 Linux CPU wheel; do not use Python 3.11+ with
this package specification. Existing root reproduction requirements contain
extra plotting/notebook/profiling tools and are not used for v3 setup.

Requirements intentionally select `torch==1.12.0+cpu` from the official CPU
index. There is no CUDA fallback; binary-only installation stops when a suitable
wheel is unavailable. NumPy/pandas/SciPy/scikit-learn wheels support glibc 2.17+
x86_64. Target system ABI/library compatibility is finally checked by imports
on the server; no Linux installation was performed during the Mac audit.

## Read-only preflight and state isolation

The implemented command is **a shell wrapper**, not a Python module:

```bash
bash experiments/odestream_fc_multidataset_v3/cluster_preflight.sh
```

It verifies exact Python and package versions/imports, Linux x86_64, Git commit,
CPU-only PyTorch, deterministic 8/8 thread configuration, all three dataset
SHA256 values using existing `provenance.verify_datasets`, and the frozen
90-online/90-scientific-warmup plan. It rejects an existing default output
directory to prevent importing Mac validation state into initial cluster setup.
It neither reads checkpoint pickle payloads nor prepares/scales data, constructs
a model, calibrates, trains, streams, or writes artifacts. It uses `python -B`.

The optional `--local-audit` mode checks Mac dependencies/data without claiming
Linux/Git/fresh-state readiness. It leaves existing Mac outputs untouched.
The wrapper was chosen because `source_manifest()` hashes every v3 `.py` file:
adding another Python module would alter the frozen source identity. No existing
scientific Python file was changed for packaging.

After initial setup the dashboard must show **0/90 scientific warmups** and
**0/90 online trajectories** complete. Never transfer `outputs/`, even though
the Mac pilot has the same protocol identity. On subsequent legitimate cluster
restarts, use status and the authorized resumable runner; the initial clean-state
preflight deliberately rejects existing cluster outputs too. Do not delete them
to make it pass. Package/library changes during a scientific run require review.

## Later execution, only after explicit authorization

Create a tmux session (`tmux new -s fc-v3`), enter the repository, activate its
Linux venv, and issue the separately authorized `run` command. tmux is optional
session persistence; checkpoints provide scientific recovery. No run command
is needed to validate this packaging. Never enable CUDA or MPS implicitly.

Missing files, SHA mismatches, wrong versions, or failed imports require
investigation. Do not download substitute datasets or reuse Mac checkpoints.
The runner independently verifies dataset hashes before scientific execution.
