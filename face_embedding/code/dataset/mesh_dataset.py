import glob
import os
from pathlib import Path
from torch.utils.data import Dataset


def parse_subject_id(path):
    stem = Path(path).stem
    return stem.split('_')[0]


class MeshDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.obj")))
        if not self.files:
            raise FileNotFoundError(f"Nessun .obj in {root_dir}")
        self.subjects = [parse_subject_id(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {"path": self.files[idx], "sid": self.subjects[idx]}
