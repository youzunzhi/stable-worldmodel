"""CLEAR-LeWM v0.3 manifest and task-success adapter.

The protocol definitions are derived from DavidSunok/CLEAR-LeWM at revision
f06b66b358f5e42aa582e4a5599d3356c29edcf4.  CLEAR-LeWM is MIT licensed.
This module keeps stable-worldmodel's planner and rollout loop, while applying
the fixed start-goal manifest and external task-completion predicates.
"""

from __future__ import annotations


CLEAR_LEWM_REVISION = 'f06b66b358f5e42aa582e4a5599d3356c29edcf4'
CLEAR_MANIFEST_SCHEMA = 'clear-lewm-manifest-v1'
CLEAR_TASKS = {'pusht', 'cube'}
CLEAR_PROTOCOLS = {'moderate', 'strict'}
CLEAR_SOLVER = {
    'batch_size': 1,
    'num_samples': 300,
    'n_steps': 30,
    'topk': 30,
}
CLEAR_CPU_THREADS = 1
