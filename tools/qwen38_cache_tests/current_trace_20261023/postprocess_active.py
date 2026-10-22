import subprocess, time
from pathlib import Path

r = Path("/results")
while (
    not (r / "capture-active.log").exists()
    or "n1 natural captured" not in (r / "capture-active.log").read_text()
):
    time.sleep(5)
time.sleep(8)
subprocess.run(["python", "/results/analyze_active.py"], check=True)
for mode in ["n0", "n1"]:
    for case in ["short", "natural"]:
        root = r / f"active-{mode}" / case
        with (root / "triage.txt").open("w") as f:
            subprocess.run(
                [
                    "python",
                    "/profile-skill/scripts/analyze_llm_torch_profile.py",
                    "--framework",
                    "sglang",
                    "--input",
                    str(root / "window.json.gz"),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
print("all active analyses passed", flush=True)
