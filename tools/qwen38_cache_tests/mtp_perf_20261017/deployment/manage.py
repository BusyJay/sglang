#!/usr/bin/env python3
"""Select and start the Compose llm/vit pair; readiness and rollback are manual."""

import argparse
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def atomic_write(path, text, mode=0o600):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Controller:
    def __init__(self, root):
        self.root = Path(root)
        self.catalog = json.loads((self.root / "models/catalog.json").read_text())
        self.state_file = self.root / "models/state.json"

    def state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        dotenv = self.root / ".env"
        match = re.search(
            r"^(?:export\s+)?LLM_PRESET\s*=\s*([^\s#]+)",
            dotenv.read_text() if dotenv.exists() else "",
            re.M,
        )
        return {
            "active": match.group(1).strip("\"'") if match else "nvfp4",
            "previous": None,
        }

    def run(self, command, capture=False, timeout=180):
        return subprocess.run(
            command,
            cwd=self.root,
            check=True,
            text=True,
            capture_output=capture,
            timeout=timeout,
        )

    def compose(self, preset, *args, capture=False):
        # Explicit file/project: never inherit another shell's COMPOSE_FILE.
        command = [
            "docker",
            "compose",
            "--project-name",
            "hermes",
            "-f",
            str(self.root / "docker-compose.yml"),
        ]
        return subprocess.run(
            command + list(args),
            cwd=self.root,
            env={**os.environ, "LLM_PRESET": preset},
            check=True,
            text=True,
            capture_output=capture,
            timeout=180,
        )

    def preflight(self, preset):
        if preset not in self.catalog:
            raise ValueError(
                f"Unknown preset {preset!r}; choose {', '.join(self.catalog)}"
            )
        spec = self.catalog[preset]
        config = json.loads(
            self.compose(preset, "config", "--format", "json", capture=True).stdout
        )
        for role in ("llm", "vit"):
            service = config["services"][role]
            image_id = self.run(
                ["docker", "image", "inspect", service["image"], "--format", "{{.Id}}"],
                capture=True,
            ).stdout.strip()
            if image_id != spec["image_id"]:
                raise ValueError(f"{preset}/{role}: image changed: {image_id}")
            mounts = {
                v["target"]: v["source"]
                for v in service["volumes"]
                if v["type"] == "bind"
            }
            for target, source in mounts.items():
                if not Path(source).exists():
                    raise ValueError(f"Missing bind source for {role}: {source}")
            if spec["source"] and mounts.get("/workspace/sglang") != spec["source"]:
                raise ValueError(f"{role}: wrong frozen SGLang source")
            if not spec["source"] and "/workspace/sglang" in mounts:
                raise ValueError("27B must use the image's original SGLang source")
            script = service["command"][1]
            if (
                not script.startswith("/prod/")
                or not Path(
                    script.replace("/prod/", "/home/jay/workspace/prod/", 1)
                ).is_file()
            ):
                raise ValueError(f"Missing launch script: {script}")
        for checkpoint in spec["checkpoints"]:
            root = Path(checkpoint)
            if not (root / "config.json").is_file():
                raise ValueError(f"Missing checkpoint config: {checkpoint}")
            index = root / "model.safetensors.index.json"
            files = (
                set(json.loads(index.read_text())["weight_map"].values())
                if index.is_file()
                else ["model.safetensors"]
            )
            for name in files:
                if (
                    Path(name).is_absolute()
                    or ".." in Path(name).parts
                    or not (root / name).is_file()
                ):
                    raise ValueError(
                        f"Missing/invalid local weight shard: {root / name}"
                    )
        if spec["pack"]:
            folder = Path(spec["pack"])
            manifest = json.loads((folder / "manifest.json").read_text())
            command = config["services"]["llm"]["command"]
            script = command[1]
            host_script = script.replace("/prod/", "/home/jay/workspace/prod/", 1)
            env = json.loads(self.run(
                ["bash", host_script, *command[2:], "--print-launch-config"], capture=True
            ).stdout)
            if (
                manifest["checkpoint_revision"] != env["CHECKPOINT_REVISION"]
                or manifest["sglang_commit"] != env["PACK_SGLANG_COMMIT"]
            ):
                raise ValueError("Checkpoint/pack provenance mismatch")
            payload = folder / manifest["payload"]
            end = max(
                t["offset"] + t["nbytes"]
                for layer in manifest["layers"].values()
                for t in layer.values()
            )
            if payload.stat().st_size != end:
                raise ValueError("Expert payload size differs from the manifest")
        print(f"Preflight passed: {preset}", flush=True)

    def persist(self, preset, state):
        path = self.root / ".env"
        text = path.read_text() if path.exists() else ""
        # Keep all other variables, including private credentials, unchanged.
        text = re.sub(r"^(?:export\s+)?LLM_PRESET\s*=.*(?:\n|$)", "", text, flags=re.M)
        if text and not text.endswith("\n"):
            text += "\n"
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        atomic_write(path, text + f"LLM_PRESET={preset}\n", mode)
        atomic_write(
            self.state_file, json.dumps(state, indent=2, ensure_ascii=False) + "\n"
        )

    def event(self, event):
        with (self.root / "models/history.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"time": timestamp(), **event}, ensure_ascii=False) + "\n"
            )

    def activate(self, preset):
        self.compose(preset, "stop", "--timeout", "60", "llm", "vit")
        self.compose(
            preset,
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--pull",
            "never",
            "--no-build",
            "llm",
            "vit",
        )

    def switch(self, target):
        old = self.state()
        previous = old["active"]
        self.preflight(target)
        state = {
            "active": target,
            "previous": previous if previous != target else old.get("previous"),
            "selected_at": timestamp(),
            "status": "requested",
        }
        # Record the selection before Docker runs so even an up failure can be
        # undone with an explicit rollback. "active" means selected, not online.
        self.persist(target, state)
        print(f"Starting {previous} -> {target}; readiness and rollback are manual", flush=True)
        try:
            self.activate(target)
        except BaseException as error:
            state["status"] = "command_failed"
            self.persist(target, state)
            self.event({"action": "switch", "from": previous, "to": target,
                        "status": "command_failed", "error": str(error)})
            print("Docker command failed; no automatic rollback. Use ./llmctl rollback if needed.",
                  file=sys.stderr, flush=True)
            raise
        state["status"] = "start_requested"
        self.persist(target, state)
        self.event({"action": "switch", "from": previous, "to": target,
                    "status": "start_requested"})
        print("Container start requested; service readiness was not checked.", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        helper = Path(__file__).resolve().parent.parent / "home-jay/remote-mapping/llm/sglang-profile.py"
        raise SystemExit(subprocess.call([sys.executable, str(helper), *sys.argv[2:]], cwd=helper.parents[3] / "models"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "fp4-local",
            "fp4-local-a4",
            "nvfp4",
            "27b",
            "switch",
            "rollback",
            "restart",
            "status",
            "check",
            "logs",
            "profile",
        ),
    )
    parser.add_argument("target", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    ctl = Controller(root)
    state = ctl.state()
    if args.command == "status":
        print(json.dumps(state, indent=2, ensure_ascii=False))
        ctl.compose(state["active"], "ps", "--all", "llm", "vit")
        return
    if args.command == "logs":
        ctl.compose(state["active"], "logs", "--tail", "100", args.target or "llm")
        return
    if args.command == "check":
        for preset in [args.target] if args.target else ctl.catalog:
            ctl.preflight(preset)
        return
    target = (
        state.get("previous")
        if args.command == "rollback"
        else state["active"]
        if args.command == "restart"
        else args.target
        if args.command == "switch"
        else args.command
    )
    if not target:
        parser.error("No target/previous preset is available")
    if args.dry_run:
        ctl.preflight(target)
        print(f"Would switch llm + vit: {state['active']} -> {target}")
        return
    with (root / "models/switch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another model switch is running")

        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("Switch interrupted")

        signal.signal(signal.SIGTERM, interrupt)
        ctl.switch(target)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"llmctl: {error}", file=sys.stderr)
        raise SystemExit(1)
