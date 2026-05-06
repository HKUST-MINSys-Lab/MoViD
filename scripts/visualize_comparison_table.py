#!/usr/bin/env python3
"""Generate SVG visualizations for the comprehensive comparison table.

The script is intentionally dependency-free so it works in lightweight paper
workspaces where matplotlib or plotting stacks may not be installed correctly.
"""

from pathlib import Path
from xml.sax.saxutils import escape
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output"

METHODS = [
    ("SOTA Pose Estimation Baselines", "HMR2.0"),
    ("SOTA Pose Estimation Baselines", "ReFit"),
    ("SOTA Pose Estimation Baselines", "HSMR"),
    ("SOTA Pose Estimation Baselines", "WHAM"),
    ("Viewpoint-Invariant Baselines", "SPIN"),
    ("Viewpoint-Invariant Baselines", "GraphCMR"),
    ("Viewpoint-Invariant Baselines", "NToP"),
    ("Our Method", "MoViD (Ours)"),
]

DATASETS = [
    ("HuMMan", "14 joints"),
    ("EMDB", "24 joints"),
    ("RICH", "24 joints"),
]

METRICS = ["PA-MPJPE", "MPJPE", "PVE"]

VALUES = {
    "HMR2.0": {
        "HuMMan": [78.5, 115.1, 132.8],
        "EMDB": [59.9, 98.7, 120.2],
        "RICH": [62.8, 103.6, 108.3],
    },
    "ReFit": {
        "HuMMan": [79.1, 117.4, 135.2],
        "EMDB": [71.2, 104.2, 123.9],
        "RICH": [67.3, 93.4, 113.4],
    },
    "HSMR": {
        "HuMMan": [65.1, 98.2, 119.3],
        "EMDB": [52.5, 92.3, 108.5],
        "RICH": [57.4, 101.2, 109.2],
    },
    "WHAM": {
        "HuMMan": [61.1, 99.7, 112.1],
        "EMDB": [52.8, 77.7, 93.6],
        "RICH": [55.7, 98.3, 111.5],
    },
    "SPIN": {
        "HuMMan": [82.6, 122.5, 141.7],
        "EMDB": [78.3, 110.9, 112.6],
        "RICH": [77.3, 113.2, 143.4],
    },
    "GraphCMR": {
        "HuMMan": [85.3, 126.8, 145.1],
        "EMDB": [86.6, 112.3, 141.5],
        "RICH": [69.5, 102.4, 113.8],
    },
    "NToP": {
        "HuMMan": [69.5, 96.9, 99.2],
        "EMDB": [74.5, 104.3, 113.6],
        "RICH": [78.3, 113.5, 123.2],
    },
    "MoViD (Ours)": {
        "HuMMan": [49.2, 80.1, 96.7],
        "EMDB": [46.7, 71.0, 87.2],
        "RICH": [51.6, 93.4, 107.8],
    },
}

PALETTE = [
    "#6E7781",
    "#8C6D31",
    "#4C78A8",
    "#72B7B2",
    "#B279A2",
    "#54A24B",
    "#F58518",
    "#D62728",
]


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text(draw, xy, text, size=18, fill="#1f2937", anchor="la", bold=False):
    draw.text(xy, str(text), font=font(size, bold), fill=fill, anchor=anchor)


def draw_centered_rotated_text(image, center, text, size=15, fill="#374151", bold=False, angle=90):
    text_font = font(size, bold)
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=text_font)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 8
    layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((tw / 2, th / 2), text, font=text_font, fill=fill, anchor="mm")
    rotated = layer.rotate(angle, expand=True)
    image.alpha_composite(rotated, (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)))


def svg_text(x, y, text, size=18, fill="#1f2937", anchor="start", weight=400, rotate=None):
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
        f'font-weight="{weight}"',
        'font-family="Arial, Helvetica, sans-serif"',
    ]
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate} {x} {y})"')
    return f'<text {" ".join(attrs)}>{escape(str(text))}</text>'


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def line(x1, y1, x2, y2, stroke="#d1d5db", sw=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"/>'


def lerp(a, b, t):
    return int(round(a + (b - a) * t))


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    a = hex_to_rgb(c1)
    b = hex_to_rgb(c2)
    return "#" + "".join(f"{lerp(a[i], b[i], t):02x}" for i in range(3))


def value_color(value, column_values):
    lo = min(column_values)
    hi = max(column_values)
    t = 0 if hi == lo else (value - lo) / (hi - lo)
    if t < 0.5:
        return mix("#D9F0E3", "#FFF2B8", t / 0.5)
    return mix("#FFF2B8", "#F2B6A8", (t - 0.5) / 0.5)


def write_heatmap():
    dataset_w = 330
    left_w = 238
    top_h = 130
    row_h = 52
    metric_w = dataset_w / 3
    width = left_w + dataset_w * len(DATASETS) + 56
    height = top_h + row_h * len(METHODS) + 118

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#ffffff"),
        svg_text(32, 42, "Overall Performance Comparison", 28, "#111827", weight=700),
        svg_text(32, 70, "Lower is better. Each metric column is color-normalized independently.", 15, "#4b5563"),
        svg_text(32, height - 24, "Values are in millimeters. Green cells indicate lower error; red cells indicate higher error.", 14, "#6b7280"),
    ]

    x0 = left_w
    y0 = top_h
    parts.append(rect(28, 92, width - 56, height - 156, "#f8fafc", "#e5e7eb", 1, 8))
    parts.append(svg_text(48, y0 - 18, "Approach", 16, "#374151", weight=700))

    for d_idx, (dataset, joints) in enumerate(DATASETS):
        dx = x0 + d_idx * dataset_w
        parts.append(svg_text(dx + dataset_w / 2, 104, f"{dataset} Dataset", 18, "#111827", "middle", 700))
        parts.append(svg_text(dx + dataset_w / 2, 126, f"({joints})", 13, "#6b7280", "middle"))
        for m_idx, metric in enumerate(METRICS):
            mx = dx + m_idx * metric_w
            parts.append(svg_text(mx + metric_w / 2, y0 - 18, metric, 13, "#374151", "middle", 700))
        if d_idx > 0:
            parts.append(line(dx, 92, dx, height - 64, "#cbd5e1", 1.2))

    all_methods = [method for _, method in METHODS]
    column_values = {}
    for dataset, _ in DATASETS:
        for metric_idx, metric in enumerate(METRICS):
            column_values[(dataset, metric)] = [VALUES[m][dataset][metric_idx] for m in all_methods]

    current_category = None
    for row_idx, (category, method) in enumerate(METHODS):
        y = y0 + row_idx * row_h
        is_ours = method.endswith("(Ours)")
        row_fill = "#fff7f7" if is_ours else ("#ffffff" if row_idx % 2 == 0 else "#f9fafb")
        parts.append(rect(28, y, width - 56, row_h, row_fill))
        parts.append(line(28, y, width - 28, y, "#e5e7eb"))

        if category != current_category:
            current_category = category
            parts.append(svg_text(48, y + 20, category.replace(" Baselines", ""), 11, "#6b7280", weight=700))

        parts.append(svg_text(48, y + 40, method, 17, "#991b1b" if is_ours else "#111827", weight=700 if is_ours else 500))

        for d_idx, (dataset, _) in enumerate(DATASETS):
            dx = x0 + d_idx * dataset_w
            for m_idx, metric in enumerate(METRICS):
                mx = dx + m_idx * metric_w
                value = VALUES[method][dataset][m_idx]
                fill = value_color(value, column_values[(dataset, metric)])
                parts.append(rect(mx + 7, y + 8, metric_w - 14, row_h - 16, fill, "#ffffff", 1, 6))
                parts.append(svg_text(mx + metric_w / 2, y + 34, f"{value:.1f}", 16, "#111827", "middle", 700 if is_ours else 500))

    parts.append(line(28, y0 + len(METHODS) * row_h, width - 28, y0 + len(METHODS) * row_h, "#cbd5e1"))
    parts.append("</svg>")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comprehensive_comparison_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def write_heatmap_png(scale=2):
    dataset_w = 330
    left_w = 238
    top_h = 130
    row_h = 52
    metric_w = dataset_w / 3
    width = int(left_w + dataset_w * len(DATASETS) + 56)
    height = int(top_h + row_h * len(METHODS) + 118)

    image = Image.new("RGBA", (width * scale, height * scale), "#ffffff")
    draw = ImageDraw.Draw(image)

    def sbox(box):
        return tuple(int(round(v * scale)) for v in box)

    def st(x, y, text, size=18, fill="#1f2937", anchor="la", bold=False):
        draw_text(draw, (int(x * scale), int(y * scale)), text, int(size * scale), fill, anchor, bold)

    def sr(x, y, w, h, fill, outline=None, radius=0, width_px=1):
        draw.rounded_rectangle(sbox((x, y, x + w, y + h)), radius=int(radius * scale), fill=fill, outline=outline, width=max(1, int(width_px * scale)))

    def sl(x1, y1, x2, y2, fill="#d1d5db", width_px=1):
        draw.line(sbox((x1, y1, x2, y2)), fill=fill, width=max(1, int(width_px * scale)))

    st(32, 42, "Overall Performance Comparison", 28, "#111827", bold=True)
    st(32, 70, "Lower is better. Each metric column is color-normalized independently.", 15, "#4b5563")
    st(32, height - 24, "Values are in millimeters. Green cells indicate lower error; red cells indicate higher error.", 14, "#6b7280")
    sr(28, 92, width - 56, height - 156, "#f8fafc", "#e5e7eb", 8)
    st(48, top_h - 18, "Approach", 16, "#374151", bold=True)

    for d_idx, (dataset, joints) in enumerate(DATASETS):
        dx = left_w + d_idx * dataset_w
        st(dx + dataset_w / 2, 104, f"{dataset} Dataset", 18, "#111827", "ma", True)
        st(dx + dataset_w / 2, 126, f"({joints})", 13, "#6b7280", "ma")
        for m_idx, metric in enumerate(METRICS):
            mx = dx + m_idx * metric_w
            st(mx + metric_w / 2, top_h - 18, metric, 13, "#374151", "ma", True)
        if d_idx > 0:
            sl(dx, 92, dx, height - 64, "#cbd5e1", 1.2)

    all_methods = [method for _, method in METHODS]
    column_values = {}
    for dataset, _ in DATASETS:
        for metric_idx, metric in enumerate(METRICS):
            column_values[(dataset, metric)] = [VALUES[m][dataset][metric_idx] for m in all_methods]

    current_category = None
    for row_idx, (category, method) in enumerate(METHODS):
        y = top_h + row_idx * row_h
        is_ours = method.endswith("(Ours)")
        row_fill = "#fff7f7" if is_ours else ("#ffffff" if row_idx % 2 == 0 else "#f9fafb")
        draw.rectangle(sbox((28, y, width - 28, y + row_h)), fill=row_fill)
        sl(28, y, width - 28, y, "#e5e7eb")
        if category != current_category:
            current_category = category
            st(48, y + 20, category.replace(" Baselines", ""), 11, "#6b7280", bold=True)
        st(48, y + 40, method, 17, "#991b1b" if is_ours else "#111827", bold=is_ours)
        for d_idx, (dataset, _) in enumerate(DATASETS):
            dx = left_w + d_idx * dataset_w
            for m_idx, metric in enumerate(METRICS):
                mx = dx + m_idx * metric_w
                value = VALUES[method][dataset][m_idx]
                fill = value_color(value, column_values[(dataset, metric)])
                sr(mx + 7, y + 8, metric_w - 14, row_h - 16, fill, "#ffffff", 6)
                st(mx + metric_w / 2, y + 34, f"{value:.1f}", 16, "#111827", "ma", is_ours)

    sl(28, top_h + len(METHODS) * row_h, width - 28, top_h + len(METHODS) * row_h, "#cbd5e1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(OUT_DIR / "comprehensive_comparison_heatmap.png", quality=95)


def write_grouped_bars():
    width = 1600
    height = 850
    margin_l = 92
    margin_r = 52
    margin_t = 112
    margin_b = 156
    panel_gap = 44
    plot_w = width - margin_l - margin_r
    panel_w = (plot_w - panel_gap * 2) / 3
    plot_h = height - margin_t - margin_b
    max_v = 150
    bar_w = 9
    metric_gap = 28
    group_w = bar_w * len(METHODS) + metric_gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#ffffff"),
        svg_text(42, 46, "Metric-wise Error Comparison", 28, "#111827", weight=700),
        svg_text(42, 74, "Lower bars are better; MoViD is highlighted in red.", 15, "#4b5563"),
    ]

    y_base = margin_t + plot_h
    for tick in range(0, 151, 25):
        y = y_base - tick / max_v * plot_h
        parts.append(line(margin_l - 10, y, width - margin_r, y, "#edf2f7"))
        parts.append(svg_text(margin_l - 18, y + 5, tick, 13, "#6b7280", "end"))
    parts.append(svg_text(32, margin_t + plot_h / 2, "Error (mm)", 15, "#374151", "middle", 700, -90))

    for d_idx, (dataset, joints) in enumerate(DATASETS):
        px = margin_l + d_idx * (panel_w + panel_gap)
        parts.append(svg_text(px + panel_w / 2, 102, f"{dataset} ({joints})", 18, "#111827", "middle", 700))
        parts.append(line(px, margin_t, px, y_base, "#cbd5e1"))
        parts.append(line(px, y_base, px + panel_w, y_base, "#9ca3af", 1.4))

        total_groups_w = group_w * 3 - metric_gap
        start_x = px + (panel_w - total_groups_w) / 2
        for m_idx, metric in enumerate(METRICS):
            gx = start_x + m_idx * group_w
            parts.append(svg_text(gx + (bar_w * len(METHODS)) / 2, y_base + 32, metric, 13, "#374151", "middle", 700))
            for method_idx, (_, method) in enumerate(METHODS):
                value = VALUES[method][dataset][m_idx]
                bh = value / max_v * plot_h
                x = gx + method_idx * bar_w
                y = y_base - bh
                fill = PALETTE[method_idx]
                parts.append(rect(x, y, bar_w - 1, bh, fill, rx=2))
                if method.endswith("(Ours)"):
                    parts.append(svg_text(x + bar_w / 2, y - 5, f"{value:.1f}", 10, "#991b1b", "middle", 700))

    legend_x = 220
    legend_y = height - 88
    legend_swatch = 13
    for idx, (_, method) in enumerate(METHODS):
        x = legend_x + idx * 154
        parts.append(rect(x, legend_y, legend_swatch, legend_swatch, PALETTE[idx], rx=2))
        # SVG text is baseline-positioned; this offset visually centers it with the swatch.
        parts.append(svg_text(x + 20, legend_y + 11, method, 13, "#374151", weight=700 if method.endswith("(Ours)") else 400))

    parts.append(svg_text(42, height - 30, "Data from the LaTeX comparison table. PA-MPJPE, MPJPE, and PVE are reported in millimeters.", 13, "#6b7280"))
    parts.append("</svg>")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comprehensive_comparison_bars.svg").write_text("\n".join(parts), encoding="utf-8")


def write_grouped_bars_png(scale=2):
    width = 1600
    height = 850
    margin_l = 92
    margin_r = 52
    margin_t = 112
    margin_b = 156
    panel_gap = 44
    plot_w = width - margin_l - margin_r
    panel_w = (plot_w - panel_gap * 2) / 3
    plot_h = height - margin_t - margin_b
    max_v = 150
    bar_w = 9
    metric_gap = 28
    group_w = bar_w * len(METHODS) + metric_gap
    y_base = margin_t + plot_h

    image = Image.new("RGBA", (width * scale, height * scale), "#ffffff")
    draw = ImageDraw.Draw(image)

    def sbox(box):
        return tuple(int(round(v * scale)) for v in box)

    def st(x, y, text, size=18, fill="#1f2937", anchor="la", bold=False):
        draw_text(draw, (int(x * scale), int(y * scale)), text, int(size * scale), fill, anchor, bold)

    def sr(x, y, w, h, fill, radius=0):
        draw.rounded_rectangle(sbox((x, y, x + w, y + h)), radius=int(radius * scale), fill=fill)

    def sl(x1, y1, x2, y2, fill="#d1d5db", width_px=1):
        draw.line(sbox((x1, y1, x2, y2)), fill=fill, width=max(1, int(width_px * scale)))

    st(42, 46, "Metric-wise Error Comparison", 28, "#111827", bold=True)
    st(42, 74, "Lower bars are better; MoViD is highlighted in red.", 15, "#4b5563")
    for tick in range(0, 151, 25):
        y = y_base - tick / max_v * plot_h
        sl(margin_l - 10, y, width - margin_r, y, "#edf2f7")
        st(margin_l - 18, y + 5, tick, 13, "#6b7280", "ra")
    draw_centered_rotated_text(image, (32 * scale, (margin_t + plot_h / 2) * scale), "Error (mm)", 15 * scale, "#374151", True, 90)

    for d_idx, (dataset, joints) in enumerate(DATASETS):
        px = margin_l + d_idx * (panel_w + panel_gap)
        st(px + panel_w / 2, 102, f"{dataset} ({joints})", 18, "#111827", "ma", True)
        sl(px, margin_t, px, y_base, "#cbd5e1")
        sl(px, y_base, px + panel_w, y_base, "#9ca3af", 1.4)
        total_groups_w = group_w * 3 - metric_gap
        start_x = px + (panel_w - total_groups_w) / 2
        for m_idx, metric in enumerate(METRICS):
            gx = start_x + m_idx * group_w
            st(gx + (bar_w * len(METHODS)) / 2, y_base + 32, metric, 13, "#374151", "ma", True)
            for method_idx, (_, method) in enumerate(METHODS):
                value = VALUES[method][dataset][m_idx]
                bh = value / max_v * plot_h
                x = gx + method_idx * bar_w
                y = y_base - bh
                sr(x, y, bar_w - 1, bh, PALETTE[method_idx], 2)
                if method.endswith("(Ours)"):
                    st(x + bar_w / 2, y - 5, f"{value:.1f}", 10, "#991b1b", "ma", True)

    legend_x = 220
    legend_y = height - 88
    legend_swatch = 13
    legend_text_size = 13
    for idx, (_, method) in enumerate(METHODS):
        x = legend_x + idx * 154
        item_center_y = legend_y + legend_swatch / 2
        sr(x, legend_y, legend_swatch, legend_swatch, PALETTE[idx], 2)
        st(
            x + 20,
            item_center_y,
            method,
            legend_text_size,
            "#374151",
            "lm",
            method.endswith("(Ours)"),
        )
    st(42, height - 30, "Data from the LaTeX comparison table. PA-MPJPE, MPJPE, and PVE are reported in millimeters.", 13, "#6b7280")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(OUT_DIR / "comprehensive_comparison_bars.png", quality=95)


def draw_hatched_bar(draw, box, fill, hatch="#111827", pattern="/", scale=2):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    bar_layer = Image.new("RGBA", (w, h), fill)
    bar_draw = ImageDraw.Draw(bar_layer)
    spacing = max(8, int(7 * scale))
    line_w = max(1, scale)
    if pattern == ".":
        step = max(10, int(9 * scale))
        radius = max(1, int(1.4 * scale))
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                bar_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=hatch)
    elif pattern == "-":
        for y in range(spacing // 2, h, spacing):
            bar_draw.line((0, y, w, y), fill=hatch, width=line_w)
    elif pattern == "|":
        for x in range(spacing // 2, w, spacing):
            bar_draw.line((x, 0, x, h), fill=hatch, width=line_w)
    elif pattern == "\\":
        for offset in range(-h, w + spacing, spacing):
            bar_draw.line((offset, 0, offset + h, h), fill=hatch, width=line_w)
    elif pattern == "x":
        for offset in range(-h, w + spacing, spacing):
            bar_draw.line((offset, h, offset + h, 0), fill=hatch, width=line_w)
        for offset in range(0, w + h + spacing, spacing):
            bar_draw.line((offset, 0, offset - h, h), fill=hatch, width=line_w)
    else:
        for offset in range(-h, w + spacing, spacing):
            bar_draw.line((offset, h, offset + h, 0), fill=hatch, width=line_w)

    draw._image.paste(bar_layer, (x1, y1))


def softened_hatch_color(fill):
    rgb = hex_to_rgb(fill)
    # Darker hatch strokes for stronger pattern contrast.
    return tuple(lerp(rgb[i], 0, 0.38) for i in range(3))


def write_reference_style_metric_png(metric_idx, metric, scale=2):
    width = 1280
    height = 780
    margin_l = 130
    margin_r = 60
    margin_t = 220
    margin_b = 110
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    datasets = [name for name, _ in DATASETS]
    methods = [method for _, method in METHODS]
    metric_values = [VALUES[method][dataset][metric_idx] for method in methods for dataset in datasets]
    raw_y_max = max(metric_values) * 1.10
    tick_step = 10 if raw_y_max <= 80 else 20
    y_max = int(math.ceil(raw_y_max / tick_step) * tick_step)
    colors = ["#E7C28D", "#65C1A8", "#FA7A43", "#A8BCE8", "#E688C6", "#9FE45C", "#F8D83E", "#D62728"]
    hatches = ["x", "-", "|", ".", ".", "/", "\\", ""]
    axis_color = "#1f2937"
    tick_color = "#1f2937"

    image = Image.new("RGBA", (width * scale, height * scale), "#ffffff")
    draw = ImageDraw.Draw(image)

    def sx(v):
        return int(round(v * scale))

    def st(x, y, text, size=18, fill="#111827", anchor="la", bold=False):
        draw_text(draw, (sx(x), sx(y)), text, sx(size), fill, anchor, bold)

    def y_pos(value):
        return sx(margin_t + plot_h - value / y_max * plot_h)

    def x_pos(value):
        return sx(value)

    # Legend box (rounded rectangle border with method swatches inside),
    # mimicking the reference figure layout.
    n_cols = 3
    n_rows = (len(methods) + n_cols - 1) // n_cols
    swatch_w = 56
    swatch_h = 22
    label_w = 176
    col_w = swatch_w + 16 + label_w
    row_h = 38
    legend_pad_x = 22
    legend_pad_y = 14
    legend_w = n_cols * col_w + 2 * legend_pad_x
    legend_h = n_rows * row_h + 2 * legend_pad_y
    legend_x0 = (width - legend_w) / 2
    legend_y0 = 24
    draw.rounded_rectangle(
        (sx(legend_x0), sx(legend_y0), sx(legend_x0 + legend_w), sx(legend_y0 + legend_h)),
        radius=sx(8), outline="#9ca3af", width=sx(2), fill="#ffffff",
    )
    for idx, method in enumerate(methods):
        col = idx % n_cols
        row = idx // n_cols
        x = legend_x0 + legend_pad_x + col * col_w
        y = legend_y0 + legend_pad_y + row * row_h

        # Vertically center each swatch and its label on the same center line.
        item_center_y = y + row_h / 2
        swatch_y = item_center_y - swatch_h / 2
        box = (sx(x), sx(swatch_y), sx(x + swatch_w), sx(swatch_y + swatch_h))
        draw_hatched_bar(draw, box, colors[idx], softened_hatch_color(colors[idx]), hatches[idx], scale)
        st(
            x + swatch_w + 12,
            item_center_y,
            method,
            20,
            "#111827",
            "lm",
            method.endswith("(Ours)"),
        )

    # Axes and gridlines.
    draw.line((sx(margin_l), sx(margin_t), sx(margin_l), sx(margin_t + plot_h)), fill=axis_color, width=sx(2))
    draw.line((sx(margin_l), sx(margin_t + plot_h), sx(margin_l + plot_w), sx(margin_t + plot_h)), fill=axis_color, width=sx(2))
    for tick in range(0, y_max + 1, tick_step):
        y = y_pos(tick)
        draw.line((sx(margin_l - 8), y, sx(margin_l), y), fill=axis_color, width=sx(2))
        if tick > 0:
            draw.line((sx(margin_l), y, sx(margin_l + plot_w), y), fill="#eef2f7", width=sx(1))
        st(margin_l - 14, y / scale + 8, tick, 22, tick_color, "ra")

    group_gap = plot_w / len(datasets)
    bar_w = 24
    bar_gap = 5
    group_bar_w = len(methods) * bar_w + (len(methods) - 1) * bar_gap

    for d_idx, dataset in enumerate(datasets):
        center = margin_l + group_gap * (d_idx + 0.5)
        start = center - group_bar_w / 2
        st(center, margin_t + plot_h + 36, dataset, 24, "#111827", "ma")
        for m_idx, method in enumerate(methods):
            value = VALUES[method][dataset][metric_idx]
            x1 = start + m_idx * (bar_w + bar_gap)
            x2 = x1 + bar_w
            y2 = margin_t + plot_h
            box = (x_pos(x1), y_pos(value), x_pos(x2), sx(y2))
            draw_hatched_bar(draw, box, colors[m_idx], softened_hatch_color(colors[m_idx]), hatches[m_idx], scale)

    st(width / 2, height - 24, "Dataset", 28, "#111827", "ma")
    draw_centered_rotated_text(
        image,
        (46 * scale, (margin_t + plot_h / 2) * scale),
        f"{metric} (mm)",
        28 * scale,
        "#111827",
        False,
        90,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"comprehensive_comparison_{metric.lower().replace('-', '_')}_reference_style.png"
    image.convert("RGB").save(OUT_DIR / filename, quality=95)


def write_reference_style_pngs():
    for idx, metric in enumerate(METRICS):
        write_reference_style_metric_png(idx, metric)


def main():
    write_heatmap()
    write_grouped_bars()
    write_heatmap_png()
    write_grouped_bars_png()
    write_reference_style_pngs()
    print("Wrote:")
    print(OUT_DIR / "comprehensive_comparison_heatmap.svg")
    print(OUT_DIR / "comprehensive_comparison_bars.svg")
    print(OUT_DIR / "comprehensive_comparison_heatmap.png")
    print(OUT_DIR / "comprehensive_comparison_bars.png")
    for metric in METRICS:
        print(OUT_DIR / f"comprehensive_comparison_{metric.lower().replace('-', '_')}_reference_style.png")


if __name__ == "__main__":
    main()
