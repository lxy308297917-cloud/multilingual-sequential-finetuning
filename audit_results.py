"""Recompute signed continual-learning metrics from the released test matrices."""
import json
import math
from pathlib import Path
from statistics import mean

path = Path(__file__).parent / "results" / "official_test_matrices.json"
matrices = json.loads(path.read_text())
for name, track in matrices.items():
    a = track["scores"]
    n = len(track["order"])
    assert len(a) == n + 1 and all(len(row) == n for row in a), name
    old = range(n - 1)
    transfer = [a[-1][i] - a[i + 1][i] for i in old]
    actual = {
        "AP": mean(a[-1]),
        "BWT": mean(transfer),
        "FWT": mean(a[i][i] - a[0][i] for i in range(1, n)),
        "Acquisition": mean(a[i + 1][i] - a[i][i] for i in range(n)),
        "Forgetting": mean(max(row[i] for row in a[i + 1:]) - a[-1][i] for i in old),
        "Negative_BWT": mean(max(0, -d) for d in transfer),
    }
    for key, value in actual.items():
        assert math.isclose(value, track["metrics"][key], abs_tol=1e-8), (name, key, value)
print(f"Verified {len(matrices)} official-test matrices and {len(matrices) * 6} metric values")
