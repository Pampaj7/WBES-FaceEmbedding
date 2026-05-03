#!/usr/bin/env python3
"""Density plots for raw-vs-registered FaceBench distances."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS = [
    ("rigid_p2p", "Rigid ICP P2P"),
    ("nicp_p2p", "NICP P2P"),
    ("nicp_p2tri", "NICP P2Tri"),
]
BLACK = (17, 24, 39)
GRAY = (107, 114, 128)
GRID = (229, 231, 235)
WHITE = (255, 255, 255)
RED = (220, 38, 38)


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
F_LABEL = font(20)
F_TICK = font(16)
F_SMALL = font(15)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rows", type=int, default=40000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def s(x: float, scale: int) -> int:
    return int(round(x * scale))


def text(draw, xy, value, fill, fnt, scale, anchor=None):
    draw.text((s(xy[0], scale), s(xy[1], scale)), value, fill=fill, font=fnt, anchor=anchor)


def line(draw, xy, fill, width, scale):
    draw.line(tuple(s(v, scale) for v in xy), fill=fill, width=max(1, width * scale))


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if not vx or not vy:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def read_rows(path: Path, max_rows: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    n = 0
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "ok") != "ok":
                continue
            try:
                row = {"raw_chamfer": float(r["raw_chamfer"])}
                for metric, _label in METRICS:
                    row[metric] = float(r[metric])
            except Exception:
                continue
            if all(math.isfinite(v) for v in row.values()):
                n += 1
                if len(rows) < max_rows:
                    rows.append(row)
                else:
                    j = rng.randrange(n)
                    if j < max_rows:
                        rows[j] = row
    return rows


def density_color(t: float) -> tuple[int, int, int]:
    # Light blue to deep indigo, with log-scaled t in [0, 1].
    stops = [(239, 246, 255), (147, 197, 253), (37, 99, 235), (30, 27, 75)]
    t = max(0.0, min(1.0, t))
    k = min(2, int(t * 3))
    a = t * 3 - k
    c0, c1 = stops[k], stops[k + 1]
    return tuple(int(c0[i] * (1 - a) + c1[i] * a) for i in range(3))


def draw_panel(draw, rows, metric, label, px, py, pw, ph, xmax, ymax, scale):
    bins_x, bins_y = 90, 70
    counts = [[0 for _ in range(bins_y)] for _ in range(bins_x)]
    by_x = [[] for _ in range(bins_x)]
    xs, ys = [], []
    for row in rows:
        x = row["raw_chamfer"]
        y = row[metric]
        if 0 <= x <= xmax and 0 <= y <= ymax:
            ix = min(bins_x - 1, int(x / xmax * bins_x))
            iy = min(bins_y - 1, int(y / ymax * bins_y))
            counts[ix][iy] += 1
            by_x[ix].append(y)
        xs.append(x)
        ys.append(y)
    max_count = max(max(col) for col in counts) or 1

    def xmap(x):
        return px + x / xmax * pw

    def ymap(y):
        return py + ph - y / ymax * ph

    for t in [0, 0.025, 0.05, 0.075, 0.10]:
        x = xmap(t)
        line(draw, (x, py, x, py + ph), GRID, 1, scale)
        text(draw, (x, py + ph + 22), f"{t:.3f}", GRAY, F_TICK, scale, "mm")
    for t in [0, ymax / 4, ymax / 2, ymax * 3 / 4, ymax]:
        y = ymap(t)
        line(draw, (px, y, px + pw, y), GRID, 1, scale)
        text(draw, (px - 8, y), f"{t:.3f}", GRAY, F_TICK, scale, "rm")

    cell_w = pw / bins_x
    cell_h = ph / bins_y
    for ix in range(bins_x):
        for iy in range(bins_y):
            c = counts[ix][iy]
            if c <= 0:
                continue
            t = math.log1p(c) / math.log1p(max_count)
            color = density_color(t)
            x0 = px + ix * cell_w
            y0 = py + ph - (iy + 1) * cell_h
            draw.rectangle(
                (s(x0, scale), s(y0, scale), s(x0 + cell_w + 0.6, scale), s(y0 + cell_h + 0.6, scale)),
                fill=(*color, 235),
            )

    m = min(xmax, ymax)
    line(draw, (xmap(0), ymap(0), xmap(m), ymap(m)), (31, 41, 55), 2, scale)

    median_pts = []
    for ix, vals in enumerate(by_x):
        if len(vals) < 20:
            continue
        x = (ix + 0.5) / bins_x * xmax
        y = quantile(vals, 0.5)
        median_pts.append((s(xmap(x), scale), s(ymap(y), scale)))
    if len(median_pts) > 1:
        draw.line(median_pts, fill=RED, width=4 * scale)

    line(draw, (px, py + ph, px + pw, py + ph), BLACK, 1, scale)
    line(draw, (px, py, px, py + ph), BLACK, 1, scale)
    text(draw, (px + pw / 2, py - 22), label, BLACK, F_LABEL, scale, "mm")
    x_std = math.sqrt(sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs))
    y_std = math.sqrt(sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys))
    note = f"r={corr(xs, ys):.2f}, std ratio={y_std / x_std:.2f}"
    text(draw, (px + 8, py + 22), note, BLACK, F_SMALL, scale)


def draw_raw_distribution_panel(draw, rows, px, py, pw, ph, xmax, ymax, scale):
    raw_vals = [r["raw_chamfer"] for r in rows]
    bins = 70
    counts = [0] * bins
    for x in raw_vals:
        if 0 <= x <= xmax:
            counts[min(bins - 1, int(x / xmax * bins))] += 1
    max_count = max(counts) or 1

    def xmap(x):
        return px + x / xmax * pw

    def ymap(y):
        return py + ph - y / max_count * ph

    for t in [0, 0.025, 0.05, 0.075, 0.10]:
        x = xmap(t)
        line(draw, (x, py, x, py + ph), GRID, 1, scale)
        text(draw, (x, py + ph + 22), f"{t:.3f}", GRAY, F_TICK, scale, "mm")
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = py + ph - frac * ph
        line(draw, (px, y, px + pw, y), GRID, 1, scale)

    points = []
    for i, c in enumerate(counts):
        x = (i + 0.5) / bins * xmax
        points.append((s(xmap(x), scale), s(ymap(c), scale)))
    draw.line(points, fill=(59, 130, 246), width=4 * scale)

    med = quantile(raw_vals, 0.5)
    q25 = quantile(raw_vals, 0.25)
    q75 = quantile(raw_vals, 0.75)
    line(draw, (xmap(med), py, xmap(med), py + ph), BLACK, 2, scale)

    line(draw, (px, py + ph, px + pw, py + ph), BLACK, 1, scale)
    line(draw, (px, py, px, py + ph), BLACK, 1, scale)
    text(draw, (px + pw / 2, py - 22), "Raw Chamfer", BLACK, F_LABEL, scale, "mm")
    text(draw, (px + 8, py + 22), "no alignment", BLACK, F_SMALL, scale)
    text(draw, (px + 8, py + 42), f"IQR={q75 - q25:.3f}", BLACK, F_SMALL, scale)


def draw_shared_axes_registered_only(rows, out_dir: Path, scale: int) -> Path:
    w, h = 1500, 660
    left, right, top, bottom, gap = 86, 42, 150, 98, 82
    pw = (w - left - right - 2 * gap) / 3
    ph = h - top - bottom
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    draw = ImageDraw.Draw(img, "RGBA")
    text(draw, (64, 34), "Raw vs registered distance density", BLACK, F_TITLE, scale)
    text(draw, (64, 68), f"Clean cross-topology evaluation, {len(rows):,} sampled mesh pairs. Red line: binned median; all panels use shared axes.", GRAY, F_SUB, scale)

    raw_vals = [r["raw_chamfer"] for r in rows]
    xmax = quantile(raw_vals, 0.995) * 1.05
    shared_ymax = xmax
    for i, (metric, label) in enumerate(METRICS):
        px = left + i * (pw + gap)
        draw_panel(draw, rows, metric, label, px, top, pw, ph, xmax, shared_ymax, scale)
    text(draw, (left + (w - left - right) / 2, h - 30), "Raw Chamfer distance", BLACK, F_LABEL, scale, "mm")
    path = out_dir / "raw_vs_registered_density_panels_shared_axes.png"
    img.save(path)
    return path


def draw_with_raw_panel(rows, out_dir: Path, scale: int) -> Path:
    w, h = 1680, 640
    left, right, top, bottom, gap = 74, 36, 145, 96, 58
    raw_pw = 270
    pw = (w - left - right - raw_pw - 3 * gap) / 3
    ph = h - top - bottom
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    draw = ImageDraw.Draw(img, "RGBA")
    text(draw, (64, 34), "Raw vs registered distance density", BLACK, F_TITLE, scale)
    text(draw, (64, 68), f"Clean cross-topology evaluation, {len(rows):,} sampled mesh pairs. Left: raw no-alignment distribution. Red line: binned median; scatter panels use shared axes.", GRAY, F_SUB, scale)

    raw_vals = [r["raw_chamfer"] for r in rows]
    xmax = quantile(raw_vals, 0.995) * 1.05
    shared_ymax = xmax
    draw_raw_distribution_panel(draw, rows, left, top, raw_pw, ph, xmax, shared_ymax, scale)
    for i, (metric, label) in enumerate(METRICS):
        px = left + raw_pw + gap + i * (pw + gap)
        draw_panel(draw, rows, metric, label, px, top, pw, ph, xmax, shared_ymax, scale)
    text(draw, (left + raw_pw + gap + (w - left - right - raw_pw - gap) / 2, h - 30), "Raw Chamfer distance", BLACK, F_LABEL, scale, "mm")

    path = out_dir / "raw_vs_registered_density_with_raw_panel.png"
    img.save(path)
    return path


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.pairs_csv), args.max_rows, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = draw_shared_axes_registered_only(rows, out_dir, args.scale)
    draw_with_raw_panel(rows, out_dir, args.scale)
    print(path)


if __name__ == "__main__":
    main()
