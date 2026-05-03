#!/usr/bin/env python3
"""Raw-vs-registered distance scatter plots for FaceBench pipeline outputs."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLACK = (17, 24, 39)
GRAY = (107, 114, 128)
GRID = (229, 231, 235)
WHITE = (255, 255, 255)
BLUE = (59, 130, 246)
GREEN = (16, 185, 129)
PURPLE = (139, 92, 246)


def font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(30, True)
F_SUB = font(19)
F_LABEL = font(21)
F_TICK = font(18)
F_SMALL = font(16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_points", type=int, default=25000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def s(v: float, scale: int) -> int:
    return int(round(v * scale))


def text(draw, xy, value, fill, fnt, scale, anchor=None):
    draw.text((s(xy[0], scale), s(xy[1], scale)), value, fill=fill, font=fnt, anchor=anchor)


def line(draw, xy, fill, width, scale):
    draw.line(tuple(s(v, scale) for v in xy), fill=fill, width=max(1, width * scale))


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def color_from_gt(gt: float, gmin: float, gmax: float) -> tuple[int, int, int]:
    if gmax <= gmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (gt - gmin) / (gmax - gmin)))
    # blue -> purple -> amber
    if t < 0.5:
        a = t / 0.5
        c0, c1 = BLUE, PURPLE
    else:
        a = (t - 0.5) / 0.5
        c0, c1 = PURPLE, (245, 158, 11)
    return tuple(int(c0[i] * (1 - a) + c1[i] * a) for i in range(3))


def reservoir_rows(path: Path, max_points: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    n = 0
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "ok") != "ok":
                continue
            n += 1
            if len(sample) < max_points:
                sample.append(row)
            else:
                j = rng.randrange(n)
                if j < max_points:
                    sample[j] = row
    return sample


def corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return math.nan
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if not vx or not vy:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def draw_scatter(rows: list[dict[str, str]], y_metric: str, y_label: str, color_by_gt: bool, out: Path, scale: int) -> None:
    data = []
    for row in rows:
        try:
            raw = float(row["raw_chamfer"])
            y = float(row[y_metric])
            gt = float(row["gt_distance"])
        except Exception:
            continue
        if math.isfinite(raw) and math.isfinite(y) and math.isfinite(gt):
            data.append((raw, y, gt, f"{row.get('topology_a','')}->{row.get('topology_b','')}"))

    w, h = 980, 620
    left, right, top, bottom = 100, 45, 86, 90
    pw, ph = w - left - right, h - top - bottom
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    d = ImageDraw.Draw(img, "RGBA")

    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    gts = [p[2] for p in data]
    xmax = max(0.1, sorted(xs)[int(0.995 * (len(xs) - 1))] * 1.05)
    ymax = max(0.05, sorted(ys)[int(0.995 * (len(ys) - 1))] * 1.05)
    gmin, gmax = min(gts), max(gts)

    def xmap(x: float) -> float:
        return left + min(x, xmax) / xmax * pw

    def ymap(y: float) -> float:
        return top + ph - min(y, ymax) / ymax * ph

    text(d, (72, 32), f"Raw Chamfer vs {y_label}", BLACK, F_TITLE, scale)
    subtitle = "Color encodes GT identity distance" if color_by_gt else "Uniform color; 25k sampled pairs from clean full-scale run"
    text(d, (72, 60), subtitle, GRAY, F_SUB, scale)

    for i in range(6):
        tx = xmax * i / 5
        ty = ymax * i / 5
        x = xmap(tx)
        y = ymap(ty)
        line(d, (x, top, x, top + ph), GRID, 1, scale)
        line(d, (left, y, left + pw, y), GRID, 1, scale)
        text(d, (x, top + ph + 26), f"{tx:.3f}", GRAY, F_TICK, scale, "mm")
        text(d, (left - 12, y), f"{ty:.3f}", GRAY, F_TICK, scale, "rm")

    line(d, (left, top + ph, left + pw, top + ph), BLACK, 1, scale)
    line(d, (left, top, left, top + ph), BLACK, 1, scale)
    text(d, (left + pw / 2, h - 28), "Raw Chamfer distance", BLACK, F_LABEL, scale, "mm")
    text(d, (22, top + ph / 2), y_label, BLACK, F_LABEL, scale, "mm")

    # y = x reference when both axes overlap.
    m = min(xmax, ymax)
    line(d, (xmap(0), ymap(0), xmap(m), ymap(m)), (31, 41, 55, 120), 2, scale)

    for raw, y, gt, _pair in data:
        color = color_from_gt(gt, gmin, gmax) if color_by_gt else GREEN
        r = 2.0
        cx, cy = xmap(raw), ymap(y)
        d.ellipse((s(cx - r, scale), s(cy - r, scale), s(cx + r, scale), s(cy + r, scale)), fill=(*color, 54))

    pear = corr(xs, ys)
    y_std = math.sqrt(sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys))
    x_std = math.sqrt(sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs))
    note = f"Pearson raw-vs-{y_metric}: {pear:.3f}    std ratio: {y_std / x_std:.3f}"
    text(d, (left + 10, top + 24), note, BLACK, F_SMALL, scale)

    if color_by_gt:
        gx, gy = w - 230, 102
        for i in range(120):
            t = i / 119
            c = color_from_gt(gmin + t * (gmax - gmin), gmin, gmax)
            d.rectangle((s(gx + i, scale), s(gy, scale), s(gx + i + 1, scale), s(gy + 14, scale)), fill=(*c, 255))
        text(d, (gx, gy + 36), f"GT low {gmin:.2f}", GRAY, F_SMALL, scale)
        text(d, (gx + 120, gy + 36), f"high {gmax:.2f}", GRAY, F_SMALL, scale, "ra")

    save(img, out)


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return math.nan
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def draw_gt_facets(rows: list[dict[str, str]], y_metric: str, y_label: str, out: Path, scale: int) -> None:
    data = []
    for row in rows:
        try:
            raw = float(row["raw_chamfer"])
            y = float(row[y_metric])
            gt = float(row["gt_distance"])
        except Exception:
            continue
        if math.isfinite(raw) and math.isfinite(y) and math.isfinite(gt):
            data.append((raw, y, gt))

    gts = [p[2] for p in data]
    q1 = quantile(gts, 1 / 3)
    q2 = quantile(gts, 2 / 3)
    groups = [
        ("Low GT", lambda g: g <= q1, (59, 130, 246)),
        ("Mid GT", lambda g: q1 < g <= q2, (139, 92, 246)),
        ("High GT", lambda g: g > q2, (245, 158, 11)),
    ]

    w, h = 1320, 520
    left, right, top, bottom = 72, 36, 96, 82
    gap = 42
    panel_w = (w - left - right - 2 * gap) / 3
    panel_h = h - top - bottom
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    d = ImageDraw.Draw(img, "RGBA")

    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    xmax = max(0.1, sorted(xs)[int(0.995 * (len(xs) - 1))] * 1.05)
    ymax = max(0.05, sorted(ys)[int(0.995 * (len(ys) - 1))] * 1.05)

    text(d, (72, 32), f"Raw Chamfer vs {y_label}, split by GT distance", BLACK, F_TITLE, scale)
    text(d, (72, 60), "Separate panels avoid color overlap; each panel uses the same axes.", GRAY, F_SUB, scale)

    for gi, (name, pred, color) in enumerate(groups):
        px = left + gi * (panel_w + gap)
        py = top

        def xmap(x: float) -> float:
            return px + min(x, xmax) / xmax * panel_w

        def ymap(y: float) -> float:
            return py + panel_h - min(y, ymax) / ymax * panel_h

        for i in range(5):
            tx = xmax * i / 4
            ty = ymax * i / 4
            x = xmap(tx)
            y = ymap(ty)
            line(d, (x, py, x, py + panel_h), GRID, 1, scale)
            line(d, (px, y, px + panel_w, y), GRID, 1, scale)
            text(d, (x, py + panel_h + 24), f"{tx:.3f}", GRAY, F_SMALL, scale, "mm")
            if gi == 0:
                text(d, (px - 10, y), f"{ty:.3f}", GRAY, F_SMALL, scale, "rm")
        line(d, (px, py + panel_h, px + panel_w, py + panel_h), BLACK, 1, scale)
        line(d, (px, py, px, py + panel_h), BLACK, 1, scale)
        text(d, (px + panel_w / 2, py - 18), name, BLACK, F_LABEL, scale, "mm")

        pts = [(raw, y, gt) for raw, y, gt in data if pred(gt)]
        for raw, y, _gt in pts:
            cx, cy = xmap(raw), ymap(y)
            r = 1.7
            d.ellipse((s(cx - r, scale), s(cy - r, scale), s(cx + r, scale), s(cy + r, scale)), fill=(*color, 42))
        mx = min(xmax, ymax)
        line(d, (xmap(0), ymap(0), xmap(mx), ymap(mx)), (31, 41, 55, 115), 2, scale)
        text(d, (px + 8, py + 22), f"n={len(pts):,}", BLACK, F_SMALL, scale)

    text(d, (left + (w - left - right) / 2, h - 26), "Raw Chamfer distance", BLACK, F_LABEL, scale, "mm")
    save(img, out)


def main() -> None:
    args = parse_args()
    rows = reservoir_rows(Path(args.pairs_csv), args.max_points, args.seed)
    out_dir = Path(args.out_dir)
    draw_scatter(rows, "rigid_p2p", "Rigid ICP P2P distance", True, out_dir / "raw_vs_rigid_icp_gtcolor.png", args.scale)
    draw_scatter(rows, "nicp_p2p", "NICP P2P distance", True, out_dir / "raw_vs_nicp_p2p_gtcolor.png", args.scale)
    draw_scatter(rows, "nicp_p2tri", "NICP P2Tri distance", True, out_dir / "raw_vs_nicp_p2tri_gtcolor.png", args.scale)
    draw_gt_facets(rows, "rigid_p2p", "Rigid ICP P2P distance", out_dir / "raw_vs_rigid_icp_gt_facets.png", args.scale)
    draw_gt_facets(rows, "nicp_p2p", "NICP P2P distance", out_dir / "raw_vs_nicp_p2p_gt_facets.png", args.scale)
    draw_gt_facets(rows, "nicp_p2tri", "NICP P2Tri distance", out_dir / "raw_vs_nicp_p2tri_gt_facets.png", args.scale)
    print(f"wrote scatter PNGs to {out_dir}")


if __name__ == "__main__":
    main()
