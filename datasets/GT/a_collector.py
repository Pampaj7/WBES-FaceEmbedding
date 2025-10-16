import os
import shutil
from pathlib import Path

# === CONFIG ===
source_dir = Path(".")
target_dir = Path("GT_BFM")
target_dir.mkdir(parents=True, exist_ok=True)

# === RACCOLTA FILE .txt ===
count = 0
for txt_file in source_dir.glob("*.txt"):
    shutil.copy(txt_file, target_dir / txt_file.name)
    count += 1

print(f"✅ Copiati {count} file in {target_dir.resolve()}")
