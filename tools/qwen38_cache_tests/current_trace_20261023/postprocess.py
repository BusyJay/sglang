import subprocess, time
from pathlib import Path

r = Path("/results")
while (
    not (r / "profile-n1/triage.txt").exists()
    or "Fuse Opportunity Table" not in (r / "profile-n1/triage.txt").read_text()
):
    time.sleep(5)
# Capture's full triage is complete; analyze serially after its serving request.
time.sleep(5)
subprocess.run(["python", "/results/crop_decode.py"], check=True)
for mode in ["n0", "n1"]:
    with (r / f"profile-{mode}/decode-triage.txt").open("w") as f:
        subprocess.run(
            [
                "python",
                "/profile-skill/scripts/analyze_llm_torch_profile.py",
                "--framework",
                "sglang",
                "--input",
                str(r / f"profile-{mode}/decode-only.json.gz"),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
subprocess.run(["python", "/results/analyze_current.py"], check=True)
