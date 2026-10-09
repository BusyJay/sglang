#!/usr/bin/env python3
"""Queue a wake-up message into a Codex session once gpu-switch finishes.

Host-side companion for environments where the workflow config has no
completion.command yet: waits for artifacts/latest-test.json to carry a
finished_at value, then queues the result path into the target thread.
Stdlib only; exits by itself after the deadline.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workset", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--deadline-hours", type=float, default=48)
    args = parser.parse_args()
    latest = Path(args.workset) / "artifacts" / "latest-test.json"
    deadline = time.monotonic() + args.deadline_hours * 3600
    while time.monotonic() < deadline:
        payload = None
        try:
            payload = json.loads(latest.read_text())
            if isinstance(payload, dict) and payload.get("finished_at"):
                break
        except Exception:
            pass
        time.sleep(args.poll_seconds)
    else:
        print(f"deadline reached without completed result: {latest}", flush=True)
        return 3
    passed = payload.get("passed")
    phase = payload.get("phase")
    errors = payload.get("errors") or []
    summary = {
        "status": "passed" if passed else "needs_review",
        "passed": bool(passed),
        "phase": phase,
        "errors": errors[:5],
        "result_path": str(latest),
    }
    message = (
        "gpu-switch 唤醒：Qwen3.8 SM120 最终验证已结束。结果摘要 "
        f"{json.dumps(summary, ensure_ascii=False)}；请读取 {latest}"
        " 与对应 stage 日志，按 .agent-workset/NEXT.md 继续验证生产恢复。"
    )
    cmd = ["codex", "queue", "--thread", args.thread, "--message", message]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(json.dumps({"rc": result.returncode, "stdout": result.stdout.strip(),
                      "stderr": result.stderr.strip()}, ensure_ascii=False))
    return 0 if result.returncode == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
