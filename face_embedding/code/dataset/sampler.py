import random


class CrossTopoPairedSampler:
    """
    Builds subject-level pairs between two mesh topologies (e.g., BFM ↔ FLAME).

    Each yielded pair corresponds to the *same subject*, but possibly different
    reconstructions or expressions, represented on two different topologies.

    Example output (for batch_size=2):
        [
            ("id0001", "BFM/id0001_001.obj", "FLAME/id0001_003.obj"),
            ("id0002", "BFM/id0002_002.obj", "FLAME/id0002_005.obj")
        ]

    This strategy encourages the network to learn topology-invariant embeddings
    that represent subject identity rather than frame-specific geometry.

    Args:
        ds_bfm: MeshDataset containing BFM meshes and subject IDs.
        ds_flame: MeshDataset containing FLAME meshes and subject IDs.
        batch_size (int): number of subjects per batch.
    """


    def __init__(self, ds_bfm, ds_flame, batch_size=2):
        # ------------------------------------------------------------
        # Build a map from subject_id → list of mesh file paths
        # for both topologies (BFM and FLAME)
        # ------------------------------------------------------------
        self.map_bfm = {}
        for p, s in zip(ds_bfm.files, ds_bfm.subjects):
            self.map_bfm.setdefault(s, []).append(p)

        self.map_flame = {}
        for p, s in zip(ds_flame.files, ds_flame.subjects):
            self.map_flame.setdefault(s, []).append(p)

        # ------------------------------------------------------------
        # Find subjects that exist in both datasets (intersection)
        # These are the only ones we can use for cross-topology pairing
        # ------------------------------------------------------------
        self.common = sorted(set(self.map_bfm.keys()) &
                             set(self.map_flame.keys()))
        if not self.common:
            raise RuntimeError(
                "No common subject_id found between BFM and FLAME.")

        self.batch_size = batch_size

    def __iter__(self):
        """
        Yields mini-batches of pairs across topologies.
        Each batch contains `batch_size` subjects,
        and for each subject it yields one BFM mesh and one FLAME mesh.
        """
        # Shuffle subject IDs before batching (ensures random order)
        ids = self.common.copy()
        random.shuffle(ids)

        # Iterate over chunks of subjects
        for i in range(0, len(ids), self.batch_size):
            chunk = ids[i:i + self.batch_size]
            if len(chunk) < self.batch_size:
                continue  # skip incomplete batches at the end

            pairs = []
            for sid in chunk:
                # Randomly pick one mesh from each topology for the same subject
                p_bfm = random.choice(self.map_bfm[sid])
                p_flm = random.choice(self.map_flame[sid])

                # Store the tuple (subject_id, path_bfm, path_flame)
                pairs.append((sid, p_bfm, p_flm))

            # Yield one batch of paired subjects
            yield pairs

    def __len__(self):
        """Return the number of complete batches that can be generated."""
        return len(self.common) // self.batch_size
