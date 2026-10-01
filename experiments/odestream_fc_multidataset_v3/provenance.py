"""Source/data integrity and machine metadata, separate from protocol identity."""
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from .config import ROOT, registry
from .checkpointing import sha, now


def source_manifest():
    here=Path(__file__).parent
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(here.glob('*.py'))}


def verify_datasets():
    for name,spec in registry().items():
        path=ROOT/spec['file']
        if not path.is_file():raise FileNotFoundError(f'{name}: required dataset absent: {spec["file"]}; no download is permitted')
        if sha(path)!=spec['sha256']:raise ValueError(f'{name}: dataset SHA256 mismatch')


def environment():
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    model=platform.processor() or 'unknown'
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):model=line.partition(':')[2].strip();break
    try:
        commit=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    try:
        import psutil
        physical=psutil.cpu_count(logical=False)
    except ImportError:physical=None
    return dict(recorded_at=now(),hostname=socket.gethostname(),cpu_model=model,
        physical_cores=physical,logical_processors=os.cpu_count(),os=platform.system(),kernel=platform.release(),
        platform=platform.platform(),python=sys.version,python_version=platform.python_version(),pytorch=str(torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,device='cpu',
        intra_threads=torch.get_num_threads(),inter_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),git_commit=commit)
