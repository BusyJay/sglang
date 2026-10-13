"""CLI for staged tuning, explicit correctness sealing and profile launch."""
import argparse
import json
import subprocess
from pathlib import Path
from .core import read, write
from .campaign import (matrix, new_plan, stage_suite, select_stage, seal_correctness,
                       make_profile, check_profile, launch_command, experimental_launch)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--suite", required=True); a.add_argument("--output", required=True); a.add_argument("--gates")
    a = sub.add_parser("stage")
    for flag in ("plan", "stage", "output"):
        a.add_argument("--" + flag, required=True)
    a.add_argument("--previous")
    a = sub.add_parser("select")
    for flag in ("plan", "stage", "results", "output"):
        a.add_argument("--" + flag, required=True)
    a.add_argument("--previous")
    a = sub.add_parser("seal-correctness")
    for flag in ("plan", "selection", "gpu-small", "replay-results", "semantic", "output"):
        a.add_argument("--" + flag, required=True)
    a = sub.add_parser("profile")
    a.add_argument("--plan", required=True); a.add_argument("--output", required=True)
    a.add_argument("--selection"); a.add_argument("--correctness")
    for name in ("check", "launch", "experiment"):
        a = sub.add_parser(name)
        for flag in (("profile", "suite", "machine", "inputs") if name != "experiment" else ("suite", "machine", "inputs")):
            a.add_argument("--" + flag, required=True)
        if name in ("launch", "experiment"):
            a.add_argument("--fallback-baseline", action="store_true")
            a.add_argument("--loader-overrides", default="{}")
            a.add_argument("--server-override", action="append", default=[])
            a.add_argument("--execute", action="store_true")
            a.add_argument("--receipt", required=True)
    args = p.parse_args(argv)
    if args.action == "plan":
        result = new_plan(matrix(read(args.suite)), read(args.gates) if args.gates else None)
    elif args.action == "stage":
        result = stage_suite(read(args.plan), args.stage, read(args.previous) if args.previous else None)
    elif args.action == "select":
        result = select_stage(read(args.plan), args.stage, args.results, read(args.previous) if args.previous else None)
    elif args.action == "seal-correctness":
        result = seal_correctness(read(args.plan), read(args.selection), args.gpu_small, args.replay_results, args.semantic)
    elif args.action == "profile":
        result = make_profile(read(args.plan), read(args.selection) if args.selection else None, read(args.correctness) if args.correctness else None)
    else:
        from .__main__ import identity
        profile = read(args.profile) if args.action != "experiment" else None
        current = identity(read(args.suite), read(args.machine), args.inputs)
        if args.action == "check":
            print(json.dumps(check_profile(profile, current), indent=2))
            return 0
        result = (experimental_launch(current, json.loads(args.loader_overrides), args.server_override)
                  if args.action == "experiment" else
                  launch_command(profile, current, args.fallback_baseline, json.loads(args.loader_overrides), args.server_override))
        write(args.receipt, result)
        print(json.dumps(result, indent=2))
        if args.execute:
            # New foreground instance only; no production service management.
            import socket
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", read(args.machine)["port"]))
            import time
            from .runner import request, stop_owned
            from .core import check_effective
            process = subprocess.Popen(result["command"], env=result["environment"], start_new_session=True)
            try:
                machine = read(args.machine)
                deadline = time.monotonic() + read(args.suite)["startup_timeout_s"]
                url = f"http://127.0.0.1:{machine['port']}"
                while True:
                    if process.poll() is not None:
                        raise ValueError("profile server exited before ready")
                    if time.monotonic() > deadline:
                        raise ValueError("profile server startup timeout")
                    try:
                        info = request(url + "/server_info", timeout=2)
                        break
                    except (OSError, TimeoutError):
                        time.sleep(0.5)
                requested = {**result["config"]["loader"], "manifest_path": machine["manifest"]}
                check_effective(info, result["config"], requested)
                if info.get("model_path") != machine["checkpoint"] or info.get("revision") != machine["model_revision"]:
                    raise ValueError("effective model/revision mismatch")
                result["effective_server_info"] = info
                write(args.receipt, result)
                return process.wait()
            finally:
                stop_owned(process)
        return 0
    write(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
