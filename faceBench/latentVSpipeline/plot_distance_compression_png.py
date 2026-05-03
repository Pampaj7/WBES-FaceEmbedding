#!/usr/bin/env python3
"""PNG plots for distance-compression analysis using only Pillow."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GEOM_METRICS = ["raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]
LABELS = {
    "raw_chamfer": "Raw Chamfer",
    "rigid_p2p": "Rigid ICP",
    "nicp_p2p": "NICP P2P",
    "nicp_p2tri": "NICP P2Tri",
}
COLORS = {
    "raw_chamfer": (59, 130, 246),
    "rigid_p2p": (245, 158, 11),
    "nicp_p2p": (16, 185, 129),
    "nicp_p2tri": (139, 92, 246),
}
BLACK = (17, 24, 39)
GRAY = (107, 114, 128)
GRID = (229, 231, 235)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
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
    p.add_argument("--clean_pairs", required=True)
    p.add_argument("--clean_summary", required=True)
    p.add_argument("--perturbed_summary", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_metric_values(path: Path, metrics: list[str]) -> dict[str, list[float]]:
    vals = {m: [] for m in metrics}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "ok") != "ok":
                continue
            for metric in metrics:
                try:
                    x = float(row[metric])
                except Exception:
                    continue
                if math.isfinite(x):
                    vals[metric].append(x)
    return vals


def canvas(w: int, h: int, scale: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w * scale, h * scale), WHITE)
    return img, ImageDraw.Draw(img)


def S(x: float, scale: int) -> int:
    return int(round(x * scale))


def line(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], fill, width: int, scale: int) -> None:
    draw.line(tuple(S(v, scale) for v in xy), fill=fill, width=max(1, width * scale))


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, fill, fnt, scale: int, anchor: str | None = None) -> None:
    draw.text((S(xy[0], scale), S(xy[1], scale)), s, fill=fill, font=fnt, anchor=anchor)


def rect(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], fill, outline, width: int, scale: int) -> None:
    draw.rectangle(tuple(S(v, scale) for v in xy), fill=fill, outline=outline, width=max(1, width * scale))


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], radius: int, fill, outline, width: int, scale: int) -> None:
    draw.rounded_rectangle(tuple(S(v, scale) for v in xy), radius=S(radius, scale), fill=fill, outline=outline, width=max(1, width * scale))


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def nice_num(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def histogram_plot(values: dict[str, list[float]], out: Path, scale: int) -> None:
    w, h = 980, 520
    left, right, top, bottom = 90, 30, 80, 80
    pw, ph = w - left - right, h - top - bottom
    xmax = 0.12
    bins = 60
    hist = {}
    ymax = 0.0
    for metric, xs in values.items():
        counts = [0] * bins
        for x in xs:
            if 0 <= x <= xmax:
                counts[min(bins - 1, int(x / xmax * bins))] += 1
        total = sum(counts) or 1
        dens = [c / total for c in counts]
        hist[metric] = dens
        ymax = max(ymax, max(dens))
    ymax *= 1.15
    img, d = canvas(w, h, scale)
    text(d, (90, 30), "Geometry distances before and after registration", BLACK, F_TITLE, scale)
    text(d, (90, 57), "Clean cross-topology pairs, 148,500 observations. Histogram area normalized per method.", GRAY, F_SUB, scale)

    def xmap(x: float) -> float:
        return left + x / xmax * pw

    def ymap(y: float) -> float:
        return top + ph - y / ymax * ph

    for t in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        x = xmap(t)
        line(d, (x, top, x, top + ph), GRID, 1, scale)
        text(d, (x, top + ph + 24), nice_num(t), GRAY, F_TICK, scale, "mm")
    line(d, (left, top + ph, left + pw, top + ph), BLACK, 1, scale)
    line(d, (left, top, left, top + ph), BLACK, 1, scale)
    text(d, (left + pw / 2, h - 24), "Distance", BLACK, F_LABEL, scale, "mm")

    for metric in GEOM_METRICS:
        pts = []
        for i, y in enumerate(hist[metric]):
            x = (i + 0.5) * xmax / bins
            pts.append((S(xmap(x), scale), S(ymap(y), scale)))
        d.line(pts, fill=COLORS[metric], width=4 * scale, joint="curve")
    lx, ly = 700, 90
    for i, metric in enumerate(GEOM_METRICS):
        y = ly + i * 26
        rect(d, (lx, y - 12, lx + 18, y + 6), COLORS[metric], COLORS[metric], 1, scale)
        text(d, (lx + 28, y - 11), LABELS[metric], BLACK, F_SMALL, scale)
    save(img, out)


def boxplot(clean_summary: list[dict[str, str]], out: Path, scale: int) -> None:
    rows = {r["metric"]: r for r in clean_summary if r["group_type"] == "overall"}
    w, h = 900, 440
    left, right, top, bottom = 180, 40, 78, 62
    pw, ph = w - left - right, h - top - bottom
    xmax = 0.12
    img, d = canvas(w, h, scale)
    text(d, (70, 30), "Registration compresses inter-subject distance spread", BLACK, F_TITLE, scale)
    text(d, (70, 57), "Boxes show P25-P75, whiskers show P05-P95, line is median.", GRAY, F_SUB, scale)

    def xmap(x: float) -> float:
        return left + x / xmax * pw

    for t in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        x = xmap(t)
        line(d, (x, top, x, top + ph), GRID, 1, scale)
        text(d, (x, top + ph + 24), nice_num(t), GRAY, F_TICK, scale, "mm")
    line(d, (left, top + ph, left + pw, top + ph), BLACK, 1, scale)
    for i, metric in enumerate(GEOM_METRICS):
        r = rows[metric]
        y = top + 48 + i * 68
        p05, p25, p50, p75, p95 = [float(r[k]) for k in ["p05", "p25", "p50", "p75", "p95"]]
        text(d, (left - 14, y), LABELS[metric], BLACK, F_LABEL, scale, "rm")
        line(d, (xmap(p05), y, xmap(p95), y), COLORS[metric], 3, scale)
        line(d, (xmap(p05), y - 10, xmap(p05), y + 10), COLORS[metric], 2, scale)
        line(d, (xmap(p95), y - 10, xmap(p95), y + 10), COLORS[metric], 2, scale)
        fill = tuple(int(c * 0.28 + 255 * 0.72) for c in COLORS[metric])
        rounded(d, (xmap(p25), y - 18, xmap(p75), y + 18), 5, fill, COLORS[metric], 2, scale)
        line(d, (xmap(p50), y - 21, xmap(p50), y + 21), BLACK, 2, scale)
    save(img, out)


def compression_bars(clean_summary: list[dict[str, str]], out: Path, scale: int) -> None:
    rows = {r["metric"]: r for r in clean_summary if r["group_type"] == "overall"}
    metrics = ["rigid_p2p", "nicp_p2p", "nicp_p2tri"]
    w, h = 760, 430
    left, right, top, bottom = 90, 40, 78, 80
    pw, ph = w - left - right, h - top - bottom
    img, d = canvas(w, h, scale)
    text(d, (70, 30), "Distance spread retained after registration", BLACK, F_TITLE, scale)
    text(d, (70, 57), "Ratio to raw Chamfer on clean full-scale evaluation. Lower means stronger compression.", GRAY, F_SUB, scale)
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + ph - t * ph
        line(d, (left, y, left + pw, y), GRID, 1, scale)
        text(d, (left - 12, y), f"{t:.2f}", GRAY, F_TICK, scale, "rm")
    bar_w, gap = 58, 95
    for i, metric in enumerate(metrics):
        cx = left + 110 + i * (bar_w * 2 + gap)
        for j, key in enumerate(["iqr_vs_raw", "std_vs_raw"]):
            val = float(rows[metric][key])
            x = cx + j * bar_w
            y = top + ph - val * ph
            color = COLORS[metric] if key == "iqr_vs_raw" else GRAY
            rounded(d, (x, y, x + bar_w - 8, top + ph), 5, color, color, 1, scale)
            text(d, (x + (bar_w - 8) / 2, y - 9), f"{val:.2f}", BLACK, F_SMALL, scale, "mm")
        text(d, (cx + bar_w - 4, h - 38), LABELS[metric], BLACK, F_SMALL, scale, "mm")
    rect(d, (565, 90, 581, 106), COLORS["rigid_p2p"], COLORS["rigid_p2p"], 1, scale)
    text(d, (590, 88), "IQR / raw IQR", BLACK, F_SMALL, scale)
    rect(d, (565, 118, 581, 134), GRAY, GRAY, 1, scale)
    text(d, (590, 116), "Std / raw std", BLACK, F_SMALL, scale)
    save(img, out)


def perturbation_lines(pert_summary: list[dict[str, str]], out: Path, scale: int) -> None:
    rows = [
        r for r in pert_summary
        if r["group_type"] == "overall"
        and r["metric"] in {"rigid_p2p", "nicp_p2p"}
        and r["noise_mode"] in {"translation", "rotation", "jitter"}
    ]
    w, h = 980, 520
    left, right, top, bottom = 80, 190, 80, 80
    pw, ph = w - left - right, h - top - bottom
    xs = [0.001, 0.0025118864, 0.0063095734, 0.0158489319, 0.0398107171, 0.1]
    xmin, xmax = min(xs), max(xs)
    img, d = canvas(w, h, scale)
    text(d, (70, 30), "Spread compression across perturbation strength", BLACK, F_TITLE, scale)
    text(d, (70, 57), "IQR ratio relative to raw Chamfer within each scenario. Raw Chamfer is always 1.0.", GRAY, F_SUB, scale)

    def xmap(x: float) -> float:
        lx = (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        return left + lx * pw

    def ymap(y: float) -> float:
        return top + ph - min(y, 1.8) / 1.8 * ph

    for t in [0, 0.5, 1.0, 1.5]:
        y = ymap(t)
        line(d, (left, y, left + pw, y), GRID, 1, scale)
        text(d, (left - 12, y), f"{t:.1f}", GRAY, F_TICK, scale, "rm")
    for x in xs:
        xp = xmap(x)
        line(d, (xp, top, xp, top + ph), GRID, 1, scale)
        text(d, (xp, top + ph + 24), f"{x:.4g}", GRAY, F_TICK, scale, "mm")
    styles = {
        ("translation", "rigid_p2p"): (COLORS["rigid_p2p"], None),
        ("rotation", "rigid_p2p"): (COLORS["rigid_p2p"], (14, 10)),
        ("jitter", "rigid_p2p"): (COLORS["rigid_p2p"], (4, 10)),
        ("translation", "nicp_p2p"): (COLORS["nicp_p2p"], None),
        ("rotation", "nicp_p2p"): (COLORS["nicp_p2p"], (14, 10)),
        ("jitter", "nicp_p2p"): (COLORS["nicp_p2p"], (4, 10)),
    }
    by: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        by.setdefault((r["noise_mode"], r["metric"]), []).append((float(r["sigma"]), float(r["iqr_vs_raw"])))
    for key, pts in sorted(by.items()):
        color, dash = styles[key]
        pts2 = [(xmap(x), ymap(y)) for x, y in sorted(pts)]
        if dash:
            for a, b in zip(pts2, pts2[1:]):
                dashed_line(d, a, b, color, 3, dash, scale)
        else:
            d.line([(S(x, scale), S(y, scale)) for x, y in pts2], fill=color, width=4 * scale)
        for x, y in pts2:
            d.ellipse((S(x - 4, scale), S(y - 4, scale), S(x + 4, scale), S(y + 4, scale)), fill=color)
    text(d, (left + pw / 2, h - 24), "Perturbation sigma, log scale", BLACK, F_LABEL, scale, "mm")
    lx, ly = left + pw + 28, 100
    legend = [("Rigid ICP", COLORS["rigid_p2p"], None), ("NICP P2P", COLORS["nicp_p2p"], None), ("translation", BLACK, None), ("rotation", BLACK, (14, 10)), ("jitter", BLACK, (4, 10))]
    for i, (lab, col, dash) in enumerate(legend):
        y = ly + i * 27
        if dash:
            dashed_line(d, (lx, y), (lx + 36, y), col, 3, dash, scale)
        else:
            line(d, (lx, y, lx + 36, y), col, 3, scale)
        text(d, (lx + 46, y - 10), lab, BLACK, F_SMALL, scale)
    save(img, out)


def distribution_panels(clean_values: dict[str, list[float]], out: Path, scale: int) -> None:
    metrics = ["raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]
    w, h = 1320, 500
    left, right, top, bottom, gap = 70, 34, 112, 78, 34
    panel_w = (w - left - right - gap * 3) / 4
    panel_h = h - top - bottom
    xmax = 0.12
    bins = 56

    hists = {}
    ymax = 0.0
    for metric in metrics:
        counts = [0] * bins
        for x in clean_values[metric]:
            if 0 <= x <= xmax:
                counts[min(bins - 1, int(x / xmax * bins))] += 1
        total = sum(counts) or 1
        dens = [c / total for c in counts]
        hists[metric] = dens
        ymax = max(ymax, max(dens))
    ymax *= 1.12

    img, d = canvas(w, h, scale)
    text(d, (64, 32), "Distance distributions before and after registration", BLACK, F_TITLE, scale)
    text(d, (64, 62), "Raw Chamfer is the no-alignment reference. Panels share the same x/y scale.", GRAY, F_SUB, scale)

    def ymap(y: float) -> float:
        return top + panel_h - y / ymax * panel_h

    for pi, metric in enumerate(metrics):
        px = left + pi * (panel_w + gap)

        def xmap(x: float) -> float:
            return px + x / xmax * panel_w

        for t in [0, 0.03, 0.06, 0.09, 0.12]:
            x = xmap(t)
            line(d, (x, top, x, top + panel_h), GRID, 1, scale)
            text(d, (x, top + panel_h + 22), f"{t:.2f}", GRAY, F_SMALL, scale, "mm")
        line(d, (px, top + panel_h, px + panel_w, top + panel_h), BLACK, 1, scale)
        line(d, (px, top, px, top + panel_h), BLACK, 1, scale)
        text(d, (px + panel_w / 2, top - 22), LABELS[metric], BLACK, F_LABEL, scale, "mm")

        pts = []
        for i, y in enumerate(hists[metric]):
            x = (i + 0.5) * xmax / bins
            pts.append((S(xmap(x), scale), S(ymap(y), scale)))
        d.line(pts, fill=COLORS[metric], width=4 * scale, joint="curve")

        vals = sorted(clean_values[metric])
        med = vals[len(vals) // 2]
        line(d, (xmap(med), top, xmap(med), top + panel_h), BLACK, 2, scale)
        text(d, (xmap(med), top + 18), f"median {med:.3f}", BLACK, F_SMALL, scale, "ma")

    text(d, (left + (w - left - right) / 2, h - 24), "Distance", BLACK, F_LABEL, scale, "mm")
    save(img, out)


def dashed_line(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], fill, width: int, dash: tuple[int, int], scale: int) -> None:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    on, off = dash
    while pos < dist:
        end = min(dist, pos + on)
        line(draw, (ax + ux * pos, ay + uy * pos, ax + ux * end, ay + uy * end), fill, width, scale)
        pos = end + off


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_values = read_metric_values(Path(args.clean_pairs), GEOM_METRICS)
    clean_summary = read_summary(Path(args.clean_summary))
    pert_summary = read_summary(Path(args.perturbed_summary))
    histogram_plot(clean_values, out_dir / "clean_geometry_distance_histogram.png", args.scale)
    boxplot(clean_summary, out_dir / "clean_geometry_distance_boxplot.png", args.scale)
    compression_bars(clean_summary, out_dir / "clean_registration_compression_ratios.png", args.scale)
    perturbation_lines(pert_summary, out_dir / "perturbation_iqr_compression_lines.png", args.scale)
    distribution_panels(clean_values, out_dir / "clean_distribution_panels_raw_to_nicp.png", args.scale)
    print(f"wrote PNGs to {out_dir}")


if __name__ == "__main__":
    main()
