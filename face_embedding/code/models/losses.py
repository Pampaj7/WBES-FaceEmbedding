import torch
import torch.nn.functional as F


def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = cosine_distance(anchor, positive)
    d_an = cosine_distance(anchor, negative)
    return torch.relu(d_ap - d_an + margin).mean()
