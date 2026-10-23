"""Single-pass reproduction of the four configurations needed for the decision."""

from run_matrix import run

for encoding, route in [
    ("native", "off"),
    ("csf", "off"),
    ("csf", "n0"),
    ("csf", "n1"),
]:
    run(f"single-{encoding}-{route}", encoding, route)
