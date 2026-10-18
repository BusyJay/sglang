"""Reproduce the frame-inspection retention bug without allocating CUDA memory."""

import gc
import json
from pathlib import Path
import types
import weakref

import torch


def load(source):
    module = types.ModuleType('audit_capture_validation')
    exec(compile(source, 'capture.py', 'exec'), module.__dict__)
    return module.inventory


def sample(capture):
    tensor = torch.empty(1024, dtype=torch.float32)
    ref = weakref.ref(tensor)
    capture()
    return ref


source = Path(__file__).with_name('capture.py').read_text()
# The serving process has imported RuntimeContext before profiling. Warm the
# same imports before checking refcount-only release with cyclic GC disabled.
fixed_capture = load(source)
sample(fixed_capture)
gc.collect()
gc.disable()
try:
    fixed = sample(fixed_capture)
    assert fixed() is None, 'Read-only diagnostic retained the tensor'
    # Restore the exact bug to demonstrate this check distinguishes it.
    broken = sample(load(source.replace('sys._getframe(2)', 'sys._getframe()')))
    assert broken() is not None, 'The original frame cycle was not reproduced'
    gc.collect()
    assert broken() is None, 'The retained tensor was not a collectable cycle'
    print(json.dumps(dict(fixed_tensor_released=True, old_cycle_reproduced=True,
                          old_cycle_released_by_gc=True)))
finally:
    gc.enable()
