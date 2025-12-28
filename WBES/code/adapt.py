import os
import shutil
from glob import glob

src_root = "/home/pampalonil/3DDFA-V3/data/cropped_3DDFAV3_neutral"
dst_root = "cropped_3DDFAV3_neutral"
os.makedirs(dst_root, exist_ok=True)

folders = [f for f in glob(os.path.join(src_root, "*")) if os.path.isdir(f)]

for folder in folders:
    txt_files = glob(os.path.join(folder, "*.txt"))
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path)
        dst_path = os.path.join(dst_root, base_name)
        shutil.copy2(txt_path, dst_path)

print(f"Done. All .txt files copied to {dst_root}")
