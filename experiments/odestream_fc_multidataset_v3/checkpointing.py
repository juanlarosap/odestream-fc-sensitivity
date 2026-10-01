"""Durable atomic persistence. Torch is imported only for scientific checkpoints."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def atomic(path,writer,binary=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb' if binary else 'w',**({} if binary else {'encoding':'utf-8'})) as f:
            writer(f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_json(path,value):atomic(path,lambda f:json.dump(value,f,indent=2,allow_nan=False))
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save_checkpoint(path,value):
    import torch
    payload=io.BytesIO();torch.save(value,payload);raw=payload.getvalue()
    atomic(path,lambda f:torch.save(dict(sha256=hashlib.sha256(raw).hexdigest(),payload=raw),f),True)


def load_checkpoint(path):
    import torch
    # Local owned checkpoints include Python/NumPy RNG and are deliberately full-state.
    try:envelope=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:envelope=torch.load(path,map_location='cpu')
    raw=envelope['payload']
    if hashlib.sha256(raw).hexdigest()!=envelope['sha256']:raise ValueError(f'Checkpoint checksum mismatch: {path}')
    try:return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    except TypeError:return torch.load(io.BytesIO(raw),map_location='cpu')
