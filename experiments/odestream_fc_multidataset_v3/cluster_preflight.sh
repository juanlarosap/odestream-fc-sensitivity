#!/usr/bin/env bash
# Read-only deployment checks. No forecasting model import or construction.
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir/../.."
python -B - "$@" <<'PY'
import argparse
import importlib
import importlib.metadata as metadata
import platform
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description='Read-only v3 cluster packaging preflight')
parser.add_argument('--local-audit', action='store_true',
                    help='Inspect Mac dependencies/data; do not certify Linux/Git/fresh-state readiness')
args = parser.parse_args()
errors = []
def check(ok, message):
    print(('PASS: ' if ok else 'FAIL: ') + message)
    if not ok:
        errors.append(message)

check(platform.python_implementation() == 'CPython' and platform.python_version() == '3.10.6',
      'CPython 3.10.6: ' + platform.python_version())
if not args.local_audit:
    check(platform.system() == 'Linux' and platform.machine() == 'x86_64',
          'Linux x86_64: ' + platform.platform())
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    check(git.returncode == 0, 'Git commit available: ' + git.stdout.strip())

requirements = Path('experiments/odestream_fc_multidataset_v3/requirements.txt')
modules = {'scikit-learn': 'sklearn', 'python-dateutil': 'dateutil',
           'typing_extensions': 'typing_extensions'}
for line in requirements.read_text().splitlines():
    if not line or line.startswith(('#', '--')):
        continue
    name, expected = line.split('==')
    if args.local_audit and name == 'torch' and platform.system() == 'Darwin':
        expected = '1.12.0'
    try:
        actual = metadata.version(name)
        importlib.import_module(modules.get(name, name))
        check(actual == expected, f'{name}: {actual}, expected {expected}')
    except (ImportError, OSError, metadata.PackageNotFoundError) as exc:
        check(False, f'{name}: {exc}')
if errors:
    sys.exit(1)

from experiments.odestream_fc_multidataset_v3 import config, datasets, run, status
from experiments.odestream_fc_multidataset_v3.provenance import verify_datasets
from experiments.odestream_fc_multidataset_v3.runtime import configure
import torch
configure()  # Thread/determinism configuration only; no tensors or models.
check(config.protocol()['execution']['device'] == 'cpu', 'Protocol device is CPU only')
check(torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 8,
      'PyTorch threads: 8 intra-op / 8 inter-op')
check(torch.are_deterministic_algorithms_enabled(), 'Deterministic algorithms enabled')
if not args.local_audit:
    check(torch.version.cuda is None, 'PyTorch CPU-only build (CUDA absent)')
try:
    verify_datasets()
    for name, spec in config.registry().items():
        print(f"PASS: {name}: {spec['file']} SHA256={spec['sha256']}")
except (OSError, ValueError) as exc:
    check(False, str(exc))
plan = config.master_plan()
check(plan['counts'] == dict(datasets=3, seeds=5, rhos=5, control_online=15,
      fc_online=75, online=90, scientific_warmups=90), 'Frozen master: 90 online / 90 scientific warmups')
selected = config.selection(plan, 'ECL', 0, 1.0)
check(sum(n['kind'] == 'online' for n in selected) == 2,
      'ECL seed0 rho1 selects paired Control + FC')
if not args.local_audit:
    check(not config.DEFAULT_OUTPUT.exists(),
          'Fresh cluster: default output directory absent (0/90 warmups and online)')
else:
    print('LOCAL AUDIT ONLY: existing pilot outputs untouched; Git/Linux/fresh-state checks omitted.')
print('No model instantiated; no dataset preparation, warmup, calibration, streaming, or artifact writes.')
sys.exit(bool(errors))
PY
