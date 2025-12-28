import os
import numpy as np

# === CONFIG ===
base_path = "/home/pampalonil/data"
methods = [
    "3DDFAv2_23470",
    "Deep3DFace_23470",
    "INORig_23470",
]

topology_report = {}

for method in methods:
    print("i see methods")
    folder = os.path.join(base_path, method)
    if not os.path.isdir(folder):
        topology_report[method] = "MISSING_FOLDER"
        continue

    try:
        # prendi il primo file valido
        print("im in try")
        candidates = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".txt") and f.startswith("id")
        ])
        if not candidates:
            topology_report[method] = "NO_TXT_FOUND"
            continue

        sample_file = os.path.join(folder, candidates[0])
        arr = np.loadtxt(sample_file)

        num_vertices = arr.shape[0]
        contains_zeros = np.any(arr == 0.0)

        topology_report[method] = f"{num_vertices} vertices, {'contains 0' if contains_zeros else 'ok'}"

    except Exception as e:
        topology_report[method] = f"ERROR: {str(e)}"

# === PRINT RESULT
print("\n=== TOPOLOGY + ZERO CHECK ===")
for method, status in topology_report.items():
    print(f"{method:<20} → {status}")
