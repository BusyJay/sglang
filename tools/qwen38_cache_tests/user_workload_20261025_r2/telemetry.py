import json
import pathlib
import subprocess
import time

out = pathlib.Path(__file__).parent
with (out / "telemetry.jsonl").open("a", buffering=1) as f:
    for _ in range(1800):
        if (out / "stop-telemetry").exists():
            break
        row = {"time": time.time()}
        for name in ["stat", "meminfo", "diskstats", "net/dev"]:
            try:
                row[name] = pathlib.Path("/proc", name).read_text()
            except OSError:
                pass
        row["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        row["processes"] = subprocess.run(
            ["docker", "top", "homelab-llm", "-eo", "pid,ppid,pcpu,rss,stat,comm"],
            capture_output=True,
            text=True,
        ).stdout
        f.write(json.dumps(row) + "\n")
        time.sleep(1)
