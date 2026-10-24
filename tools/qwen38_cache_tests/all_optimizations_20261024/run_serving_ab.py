"""Run a baseline, single-switch screen, or short startup-default validation.

Inside qwen38-csf-defaults-20261023, select baseline, hc, cta, mxfp8, or defaults.
Outputs go to the writable /results mount; each invocation shuts its server down.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

variants = {
    "baseline": "000",
    "baseline_repeat": "000",
    "hc": "100",
    "cta": "010",
    "mxfp8": "001",
    "hc_cta": "110",
    "all": "111",
    "defaults": None,
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("variant", choices=variants)
parser.add_argument("--reference", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
variant = args.variant
root = Path("/results")
source = Path(__file__).resolve().parent
output = args.output or root / "all_optimizations_20261024" / variant
output.mkdir(parents=True, exist_ok=False)
environment = json.loads((source / "serving-environment.json").read_text())
environment.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    ROOT_CSF_CHAT_AFTER="5",
)
reference = args.reference or source / "serving-command.json"
launch = json.loads(reference.read_text())
environment.update(launch.get("environment", {}))
switches = (
    "SGLANG_OPT_HC_MIX_SPLIT_K",
    "SGLANG_OPT_QWEN38_FILL_CTA_TABLE",
    "SGLANG_OPT_QWEN38_MXFP8_SMALL_M",
)
if variants[variant] is None:
    for name in switches:
        environment.pop(name, None)
        os.environ.pop(name, None)
else:
    environment.update(zip(switches, variants[variant]))
os.environ.update(environment)
(output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(output / "command.json").write_text(reference.read_text())
from sglang.srt.environ import envs

(output / "effective-switches.json").write_text(
    json.dumps({name: getattr(envs, name).get() for name in switches}, indent=2) + "\n"
)
sys.argv = [
    __file__,
    "--reference",
    str(reference),
    "--output",
    str(output),
    "--modes",
    "default",
]
if args.quick:
    sys.argv.append("--short-only")
spec = importlib.util.spec_from_file_location(
    "checked_ab", root / "benchmark_checked.py"
)
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)
try:
    checked.benchmark.main()
except Exception as error:
    destination = output / "performance.json"
    report = json.loads(destination.read_text()) if destination.exists() else {}
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise
