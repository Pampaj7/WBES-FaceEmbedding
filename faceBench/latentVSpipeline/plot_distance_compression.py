#!/usr/bin/env python3
"""Dependency-free SVG plots for distance-compression analysis."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


GEOM_METRICS = ["raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]
LABELS = {
    "raw_chamfer": "Raw Chamfer",
    "rigid_p2p": "Rigid ICP",
    "nicp_p2p": "NICP P2P",
    "nicp_p2tri": "NICP P2Tri",
}
COLORS = {
    "raw_chamfer": "#3B82F6",
    "rigid_p2p": "#F59E0B",
    "nicp_p2p": "#10B981",
    "nicp_p2tri": "#8B5CF6",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_pairs", required=True)
    p.add_argument("--clean_summary", required=True)
    p.add_argument("--perturbed_summary", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(path: Path, body: str, w: int, h: int) -> None:
    css = """
    <style>
      .bg { fill: #ffffff; }
      .axis { stroke: #111827; stroke-width: 1; }
      .grid { stroke: #E5E7EB; stroke-width: 1; }
      .tick { fill: #374151; font: 12px Arial, sans-serif; }
      .label { fill: #111827; font: 13px Arial, sans-serif; }
      .title { fill: #111827; font: bold 18px Arial, sans-serif; }
      .sub { fill: #6B7280; font: 12px Arial, sans-serif; }
    </style>
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'{css}<rect class="bg" width="{w}" height="{h}"/>\n{body}\n</svg>\n'
    )


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


def nice_num(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def histogram_plot(values: dict[str, list[float]], out: Path) -> None:
    w, h = 980, 520
    left, right, top, bottom = 90, 30, 70, 80
    pw, ph = w - left - right, h - top - bottom
    xmax = 0.12
    bins = 60
    ymax = 0.0
    hist = {}
    for metric, xs in values.items():
        counts = [0] * bins
        for x in xs:
            if 0 <= x <= xmax:
                idx = min(bins - 1, int(x / xmax * bins))
                counts[idx] += 1
        total = sum(counts) or 1
        dens = [c / total for c in counts]
        ymax = max(ymax, max(dens))
        hist[metric] = dens
    ymax *= 1.15

    def xmap(x: float) -> float:
        return left + x / xmax * pw

    def ymap(y: float) -> float:
        return top + ph - y / ymax * ph

    parts = [
        '<text class="title" x="90" y="34">Geometry distances before and after registration</text>',
        '<text class="sub" x="90" y="54">Clean cross-topology pairs, 148,500 observations. Histogram area normalized per method.</text>',
    ]
    for t in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        x = xmap(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top+ph+24}" text-anchor="middle">{nice_num(t)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    parts.append(f'<text class="label" x="{left+pw/2:.1f}" y="{h-24}" text-anchor="middle">Distance</text>')
    parts.append(f'<text class="label" x="22" y="{top+ph/2:.1f}" transform="rotate(-90 22 {top+ph/2:.1f})" text-anchor="middle">Normalized count</text>')

    for metric in GEOM_METRICS:
        pts = []
        for i, y in enumerate(hist[metric]):
            x = (i + 0.5) * xmax / bins
            pts.append(f"{xmap(x):.1f},{ymap(y):.1f}")
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{COLORS[metric]}" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    lx, ly = 700, 82
    for i, metric in enumerate(GEOM_METRICS):
        y = ly + i * 24
        parts.append(f'<rect x="{lx}" y="{y-11}" width="16" height="16" rx="2" fill="{COLORS[metric]}"/>')
        parts.append(f'<text class="label" x="{lx+24}" y="{y+2}">{esc(LABELS[metric])}</text>')
    write(out, "\n".join(parts), w, h)


def boxplot(clean_summary: list[dict[str, str]], out: Path) -> None:
    rows = {r["metric"]: r for r in clean_summary if r["group_type"] == "overall"}
    w, h = 900, 440
    left, right, top, bottom = 170, 40, 70, 60
    pw, ph = w - left - right, h - top - bottom
    xmax = 0.12

    def xmap(x: float) -> float:
        return left + x / xmax * pw

    parts = []
    for t in [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        x = xmap(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top+ph+24}" text-anchor="middle">{nice_num(t)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    for i, metric in enumerate(GEOM_METRICS):
        r = rows[metric]
        y = top + 50 + i * 68
        p05, p25, p50, p75, p95 = [float(r[k]) for k in ["p05", "p25", "p50", "p75", "p95"]]
        parts.append(f'<text class="label" x="{left-14}" y="{y+5}" text-anchor="end">{esc(LABELS[metric])}</text>')
        parts.append(f'<line x1="{xmap(p05):.1f}" y1="{y}" x2="{xmap(p95):.1f}" y2="{y}" stroke="{COLORS[metric]}" stroke-width="3"/>')
        parts.append(f'<line x1="{xmap(p05):.1f}" y1="{y-9}" x2="{xmap(p05):.1f}" y2="{y+9}" stroke="{COLORS[metric]}" stroke-width="2"/>')
        parts.append(f'<line x1="{xmap(p95):.1f}" y1="{y-9}" x2="{xmap(p95):.1f}" y2="{y+9}" stroke="{COLORS[metric]}" stroke-width="2"/>')
        parts.append(f'<rect x="{xmap(p25):.1f}" y="{y-17}" width="{xmap(p75)-xmap(p25):.1f}" height="34" rx="4" fill="{COLORS[metric]}" opacity="0.28" stroke="{COLORS[metric]}" stroke-width="2"/>')
        parts.append(f'<line x1="{xmap(p50):.1f}" y1="{y-19}" x2="{xmap(p50):.1f}" y2="{y+19}" stroke="#111827" stroke-width="2"/>')
    write(out, "\n".join(parts), w, h)


def compression_bars(clean_summary: list[dict[str, str]], out: Path) -> None:
    rows = {r["metric"]: r for r in clean_summary if r["group_type"] == "overall"}
    metrics = ["rigid_p2p", "nicp_p2p", "nicp_p2tri"]
    w, h = 760, 430
    left, right, top, bottom = 90, 40, 70, 80
    pw, ph = w - left - right, h - top - bottom
    parts = [
        '<text class="title" x="70" y="34">Distance spread retained after registration</text>',
        '<text class="sub" x="70" y="54">Ratio to raw Chamfer on clean full-scale evaluation. Lower means stronger compression.</text>',
    ]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + ph - t * ph
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{t:.2f}</text>')
    bar_w = 58
    gap = 95
    for i, metric in enumerate(metrics):
        cx = left + 110 + i * (bar_w * 2 + gap)
        for j, key in enumerate(["iqr_vs_raw", "std_vs_raw"]):
            val = float(rows[metric][key])
            x = cx + j * bar_w
            y = top + ph - val * ph
            color = COLORS[metric] if key == "iqr_vs_raw" else "#6B7280"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-8}" height="{val*ph:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text class="tick" x="{x+(bar_w-8)/2:.1f}" y="{y-7:.1f}" text-anchor="middle">{val:.2f}</text>')
        parts.append(f'<text class="label" x="{cx+bar_w-4:.1f}" y="{h-38}" text-anchor="middle">{esc(LABELS[metric])}</text>')
    parts.append(f'<rect x="{w-195}" y="90" width="14" height="14" fill="#F59E0B"/><text class="tick" x="{w-174}" y="102">IQR / raw IQR</text>')
    parts.append(f'<rect x="{w-195}" y="114" width="14" height="14" fill="#6B7280"/><text class="tick" x="{w-174}" y="126">Std / raw std</text>')
    write(out, "\n".join(parts), w, h)


def perturbation_lines(pert_summary: list[dict[str, str]], out: Path) -> None:
    keep = {"raw_chamfer", "rigid_p2p", "nicp_p2p"}
    rows = [
        r for r in pert_summary
        if r["group_type"] == "overall" and r["metric"] in keep and r["noise_mode"] in {"translation", "rotation", "jitter"}
    ]
    w, h = 980, 520
    left, right, top, bottom = 80, 190, 70, 80
    pw, ph = w - left - right, h - top - bottom
    xs = [0.001, 0.0025118864, 0.0063095734, 0.0158489319, 0.0398107171, 0.1]
    xmin, xmax = min(xs), max(xs)

    def xmap(x: float) -> float:
        lx = (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        return left + lx * pw

    def ymap(y: float) -> float:
        return top + ph - min(y, 1.8) / 1.8 * ph

    parts = [
        '<text class="title" x="70" y="34">Spread compression across perturbation strength</text>',
        '<text class="sub" x="70" y="54">IQR ratio relative to raw Chamfer within each scenario. Raw Chamfer is always 1.0.</text>',
    ]
    for t in [0, 0.5, 1.0, 1.5]:
        y = ymap(t)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{t:.1f}</text>')
    for x in xs:
        xp = xmap(x)
        parts.append(f'<line class="grid" x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top+ph}"/>')
        parts.append(f'<text class="tick" x="{xp:.1f}" y="{top+ph+24}" text-anchor="middle">{x:.4g}</text>')
    styles = {
        ("translation", "rigid_p2p"): ("#F59E0B", ""),
        ("rotation", "rigid_p2p"): ("#F59E0B", "6 5"),
        ("jitter", "rigid_p2p"): ("#F59E0B", "2 5"),
        ("translation", "nicp_p2p"): ("#10B981", ""),
        ("rotation", "nicp_p2p"): ("#10B981", "6 5"),
        ("jitter", "nicp_p2p"): ("#10B981", "2 5"),
    }
    by = {}
    for r in rows:
        if r["metric"] == "raw_chamfer":
            continue
        by.setdefault((r["noise_mode"], r["metric"]), []).append((float(r["sigma"]), float(r["iqr_vs_raw"])))
    for key, pts in sorted(by.items()):
        color, dash = styles[key]
        pts = sorted(pts)
        pstr = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{pstr}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>')
        for x, y in pts:
            parts.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="4" fill="{color}"/>')
    parts.append(f'<text class="label" x="{left+pw/2:.1f}" y="{h-24}" text-anchor="middle">Perturbation sigma, log scale</text>')
    parts.append(f'<text class="label" x="22" y="{top+ph/2:.1f}" transform="rotate(-90 22 {top+ph/2:.1f})" text-anchor="middle">IQR / raw Chamfer IQR</text>')
    lx, ly = left + pw + 28, 96
    legend = [
        ("Rigid ICP", "#F59E0B", ""),
        ("NICP P2P", "#10B981", ""),
        ("translation", "#111827", ""),
        ("rotation", "#111827", "6 5"),
        ("jitter", "#111827", "2 5"),
    ]
    for i, (label, color, dash) in enumerate(legend):
        y = ly + i * 26
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+32}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        parts.append(f'<text class="tick" x="{lx+42}" y="{y+4}">{esc(label)}</text>')
    write(out, "\n".join(parts), w, h)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_values = read_metric_values(Path(args.clean_pairs), GEOM_METRICS)
    clean_summary = read_summary(Path(args.clean_summary))
    pert_summary = read_summary(Path(args.perturbed_summary))
    histogram_plot(clean_values, out_dir / "clean_geometry_distance_histogram.svg")
    boxplot(clean_summary, out_dir / "clean_geometry_distance_boxplot.svg")
    compression_bars(clean_summary, out_dir / "clean_registration_compression_ratios.svg")
    perturbation_lines(pert_summary, out_dir / "perturbation_iqr_compression_lines.svg")
    print(f"wrote SVGs to {out_dir}")


if __name__ == "__main__":
    main()
