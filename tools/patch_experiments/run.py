#!/usr/bin/env python3
"""Generic manifest-driven Docker patch experiment runner (Python stdlib only)."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def bundle_file(bundle, name):
    path = (bundle / name).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError(f"Invalid bundle file: {name}")
    return path


def verify_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    for name in manifest["patches"]:
        bundle_file(bundle, name)
    return manifest


def patch_command(repo, manifest, bundle, reverse=False, check=False):
    paths = [str(bundle_file(bundle, name)) for name in manifest["patches"]]
    command = ["git", "apply", "--whitespace=error"]
    if reverse:
        command.append("--reverse")
        paths.reverse()
    if check:
        command.append("--check")
    return subprocess.run(
        command + paths, cwd=repo, capture_output=True, text=True, check=False
    )


def source_state(repo, manifest, bundle):
    if patch_command(repo, manifest, bundle, check=True).returncode == 0:
        return "base"
    if patch_command(repo, manifest, bundle, reverse=True, check=True).returncode == 0:
        return "patched"
    raise RuntimeError(
        "Patch cannot be applied or reversed cleanly; inspect local changes"
    )


def patch(repo, manifest, bundle, reverse=False):
    wanted = "base" if reverse else "patched"
    if source_state(repo, manifest, bundle) == wanted:
        print(f"Source already {wanted}; no changes")
        return
    result = patch_command(repo, manifest, bundle, reverse=reverse)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    print(f"Source changed to {wanted}; no services restarted")


def render(argv, context):
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(a, str) for a in argv)
    ):
        raise ValueError("Step argv must be a nonempty string array")
    # Replace named tokens only; literal braces in Python/JSON code are preserved.
    result = []
    for argument in argv:
        for key, value in context.items():
            argument = argument.replace("{" + key + "}", str(value))
        result.append(argument)
    return result


def logged(command, cwd, log, timeout, env=None):
    print("Running:", " ".join(command), flush=True)
    with log.open("w") as stream:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return process.returncode


def docker_command(args, repo, bundle, output, command, container_name, environment):
    result = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--init",
        "--workdir",
        str(repo),
        "--shm-size",
        args.shm_size,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={repo},dst={repo},readonly",
        "--mount",
        f"type=bind,src={bundle},dst={bundle},readonly",
        "--mount",
        f"type=bind,src={output},dst={output}",
        "--entrypoint",
        args.python,
    ]
    if args.gpus != "none":
        result += ["--gpus", args.gpus]
    for mount in args.mount:
        result += ["--mount", mount]
    for volume in args.volume:
        result += ["--volume", volume]
    for key, value in environment.items():
        result += ["--env", f"{key}={value}"]
    result += [args.image] + command[1:]
    if command[0] != args.python:
        raise ValueError(
            "Container steps must start with {python}; entrypoint is overridden"
        )
    return result


def remove_container(name):
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def experiment(args, repo, bundle, output, manifest):
    if args.rounds < 3 or args.iterations < 5:
        raise ValueError("Use rounds >= 3 and iterations >= 5")
    if not args.image:
        raise ValueError("Docker experiment requires --image IMAGE_NAME")
    inspected = subprocess.check_output(
        ["docker", "image", "inspect", args.image], text=True
    )
    image_info = json.loads(inspected)[0]
    context = {
        "repo": repo,
        "bundle": bundle,
        "output": output,
        "python": args.python,
        "iterations": args.iterations,
        "rounds": args.rounds,
    }
    context.update(manifest.get("parameters", {}))
    environment = {
        key: render([value], context)[0]
        for key, value in manifest.get("container_env", {}).items()
    }
    environment.update(
        PYTHONPATH=str(repo / "python"),
        PYTHONUNBUFFERED="1",
    )
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError("--env requires KEY=VALUE")
        environment[key] = value
    # Source selection is invariant even when custom environment variables are supplied.
    environment["PYTHONPATH"] = str(repo / "python")
    execution = {
        "image": args.image,
        "image_id": image_info["Id"],
        "repo": str(repo),
        "bundle": str(bundle),
        "base_commit": manifest.get("base_commit"),
        "mounts": args.mount,
        "volumes": args.volume,
        "started_at": time.time(),
        "steps": [],
    }
    write_json(output / "execution.json", execution)
    for step in manifest["steps"]:
        name = step["name"]
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid step name: {name}")
        command = render(step["argv"], context)
        container_name = "patch-exp-" + uuid.uuid4().hex[:12]
        docker = docker_command(
            args, repo, bundle, output, command, container_name, environment
        )
        try:
            code = logged(
                docker,
                repo,
                output / (name + ".log"),
                step.get("timeout_seconds", args.timeout),
            )
        finally:
            # Also clean up after timeout/interrupt. --rm cleans normal completion.
            remove_container(container_name)
        execution["steps"].append({"name": name, "argv": command, "exit_code": code})
        write_json(output / "execution.json", execution)
        if code:
            raise RuntimeError(
                f"Step {name} failed ({code}); see {output / (name + '.log')}"
            )
    execution["finished_at"] = time.time()
    write_json(output / "execution.json", execution)


def assess(args, repo, bundle, output, manifest):
    rule = manifest["assessment"]
    context = {
        "repo": repo,
        "bundle": bundle,
        "output": output,
        "python": sys.executable,
        "report": args.report or output / rule["report"],
        "min_speedup": args.min_speedup,
        "max_p95_ratio": args.max_p95_ratio,
        "max_scratch_mib": args.max_scratch_mib,
    }
    code = logged(
        render(rule["argv"], context), repo, output / "assessment.log", args.timeout
    )
    result = (
        read_json(output / "result.json") if (output / "result.json").exists() else None
    )
    if code not in (0, 2) or result is None:
        raise RuntimeError(f"Assessment failed ({code}); see assessment.log")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["all", "apply", "experiment", "assess", "rollback"]
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--image", help="Existing Docker image name/tag; never pulled automatically"
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help="Docker --gpus value, e.g. device=0; none for CPU bundles",
    )
    parser.add_argument(
        "--python", default="python3", help="Python executable inside Docker image"
    )
    parser.add_argument(
        "--env", action="append", default=[], help="Container KEY=VALUE; repeatable"
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Additional Docker --mount specification; repeatable",
    )
    parser.add_argument(
        "--volume",
        "-v",
        action="append",
        default=[],
        help="Additional Docker -v specification; repeatable",
    )
    parser.add_argument("--shm-size", default="2g")
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-step timeout seconds"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--max-p95-ratio", type=float, default=1.05)
    parser.add_argument("--max-scratch-mib", type=float, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo, bundle = args.repo.resolve(), args.bundle.resolve()
    output = (
        args.output or bundle / "results" / time.strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifest = verify_bundle(bundle)
        if args.action == "all" and not args.image:
            raise ValueError("all requires --image IMAGE_NAME")
        if args.action in ("all", "apply", "rollback"):
            patch(repo, manifest, bundle, reverse=args.action == "rollback")
        if args.action in ("all", "experiment"):
            if source_state(repo, manifest, bundle) != "patched":
                raise ValueError("Apply patch first, or use all")
            experiment(args, repo, bundle, output, manifest)
        if args.action in ("all", "experiment", "assess"):
            return assess(args, repo, bundle, output, manifest)
        write_json(
            output / "result.json",
            {
                "status": "PATCHED" if args.action == "apply" else "ROLLED_BACK",
                "production_ready": False,
            },
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- preserve failure evidence
        result = {"status": "ERROR", "production_ready": False, "error": str(exc)}
        write_json(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        print(f"Failure evidence: {output}; no services restarted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
