"""Small deterministic I/O and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, target)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as stream:
        return json.load(stream)


def git_provenance(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd)

    def run(*args: str) -> str:
        return subprocess.check_output(
            ['git', '-C', str(root), *args], text=True
        ).strip()

    status = run('status', '--porcelain=v1')
    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('branch', '--show-current'),
        'dirty': bool(status),
        'status_porcelain': status.splitlines(),
    }


def software_versions() -> dict[str, str]:
    versions = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
    }
    for name in ('numpy', 'torch', 'torchvision', 'h5py', 'pyarrow'):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, '__version__', 'unknown'))
    return versions
