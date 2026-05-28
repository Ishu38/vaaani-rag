"""Multi-type diagram renderer for LLM-emitted specs.

Supports five marker types the LLM can emit inline:

  [[PLOT:{"expr":"sin(x)","x_min":-3.14,"x_max":3.14}]]
  [[CHART:{"type":"bar","labels":["A","B"],"values":[3,7]}]]
  [[DOT:{"graph":"digraph { A -> B; B -> C; }"}]]
  [[CIRCUIT:{"elements":[...]}]]
  [[GEOM:{"type":"triangle","vertices":[[0,0],[4,0],[2,3]]}]]

All rendered server-side to PNG in the data/figures/ directory with a
dark theme matching the chat UI. Every input is validated — no eval/exec.
"""
from __future__ import annotations

import io
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from config import DATA_DIR

FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#131318"
DARK_GRID = "#23232c"
DARK_SPINE = "#3a3a48"
DARK_TICK = "#9a9aa8"
LIGHT_TEXT = "#ececf1"
ACCENT = "#e63b66"
ACCENT_BLUE = "#4a9eff"
ACCENT_GREEN = "#4ade80"
ACCENT_YELLOW = "#facc15"

SYMPY_LOCALS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs,
    "pi": sp.pi, "E": sp.E, "e": sp.E,
}
MAX_POINTS = 2000
MIN_RANGE_SPAN = 1e-6
MAX_RANGE_SPAN = 1e4


@dataclass
class RenderedFigure:
    id: str
    path: Path
    url: str
    caption: str
    expr: str = ""


class DiagramError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Dark theme helper
# ═══════════════════════════════════════════════════════════════════════

def _dark_axes(ax, title="", x_label="", y_label="", grid=True):
    fig = ax.figure
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(DARK_SPINE)
    ax.tick_params(colors=DARK_TICK, labelsize=9)
    ax.xaxis.label.set_color(LIGHT_TEXT)
    ax.yaxis.label.set_color(LIGHT_TEXT)
    ax.title.set_color(LIGHT_TEXT)
    if grid:
        ax.grid(True, color=DARK_GRID, linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def _save_figure(fig, caption="", expr="") -> RenderedFigure:
    fig_id = uuid.uuid4().hex[:12]
    out_path = FIGURES_DIR / f"{fig_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    return RenderedFigure(
        id=fig_id,
        path=out_path,
        url=f"/figures/{fig_id}.png",
        caption=caption[:160],
        expr=expr,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. PLOT — function graphs (SymPy + matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def render_plot(spec: dict) -> RenderedFigure:
    expr_str = (spec.get("expr") or "").strip()
    if not expr_str or len(expr_str) > 200:
        raise DiagramError("expr is empty or too long")
    try:
        x = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={**SYMPY_LOCALS, "x": x})
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise DiagramError(f"could not parse expression: {e}")
    free = expr.free_symbols
    if free and free != {x}:
        raise DiagramError(f"expression must depend only on x; saw {sorted(map(str, free))}")
    fn = sp.lambdify(x, expr, modules=["numpy"])

    try:
        x_min = float(spec.get("x_min", -10))
        x_max = float(spec.get("x_max", 10))
    except (TypeError, ValueError):
        raise DiagramError("x_min/x_max must be numbers")
    if not (x_min < x_max):
        raise DiagramError("x_min must be < x_max")
    span = x_max - x_min
    if span < MIN_RANGE_SPAN or span > MAX_RANGE_SPAN:
        raise DiagramError(f"range span out of bounds")

    title = (spec.get("title") or f"y = {sp.pretty(expr, use_unicode=False)}")[:160]
    x_label = (spec.get("x_label") or "x")[:60]
    y_label = (spec.get("y_label") or "f(x)")[:60]
    mark_zero = bool(spec.get("mark_zero", True))
    fill_area = spec.get("fill_between")  # optional: [x_start, x_end] for area fill

    xs = np.linspace(x_min, x_max, MAX_POINTS)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ys = fn(xs)
    ys = np.where(np.isfinite(ys), ys, np.nan)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    _dark_axes(ax, title, x_label, y_label)

    if mark_zero:
        ax.axhline(0, color=DARK_SPINE, linewidth=0.8)
        ax.axvline(0, color=DARK_SPINE, linewidth=0.8)

    ax.plot(xs, ys, color=ACCENT, linewidth=2.0)

    # Area fill between curve and x-axis
    if fill_area and isinstance(fill_area, list) and len(fill_area) == 2:
        try:
            fa, fb = float(fill_area[0]), float(fill_area[1])
            mask = (xs >= fa) & (xs <= fb)
            ax.fill_between(xs[mask], 0, ys[mask], color=ACCENT, alpha=0.2)
        except (TypeError, ValueError):
            pass

    # Second curve overlay
    expr2_str = (spec.get("expr2") or "").strip()
    if expr2_str:
        try:
            expr2 = sp.sympify(expr2_str, locals={**SYMPY_LOCALS, "x": x})
            fn2 = sp.lambdify(x, expr2, modules=["numpy"])
            ys2 = fn2(xs)
            ys2 = np.where(np.isfinite(ys2), ys2, np.nan)
            ax.plot(xs, ys2, color=ACCENT_BLUE, linewidth=2.0, linestyle="--")
        except Exception:
            pass

    return _save_figure(fig, title, expr_str)


# ═══════════════════════════════════════════════════════════════════════
# 2. CHART — statistical charts (matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def render_chart(spec: dict) -> RenderedFigure:
    chart_type = (spec.get("type") or "bar").lower()
    title = (spec.get("title") or "").strip()[:160]
    x_label = (spec.get("x_label") or "").strip()[:60]
    y_label = (spec.get("y_label") or "").strip()[:60]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    _dark_axes(ax, title, x_label, y_label)

    valid_types = {"bar", "histogram", "hist", "scatter", "boxplot", "box", "pie"}
    if chart_type not in valid_types:
        raise DiagramError(f"unknown chart type '{chart_type}'; allowed: {sorted(valid_types)}")

    if chart_type == "pie":
        labels = spec.get("labels") or []
        values = spec.get("values") or []
        if not values or len(values) < 1:
            raise DiagramError("pie chart needs 'values' array")
        colors = [ACCENT, ACCENT_BLUE, ACCENT_GREEN, ACCENT_YELLOW,
                  "#f472b6", "#a78bfa", "#fb923c", "#2dd4bf"]
        wedges, texts = ax.pie(values, labels=labels[:len(values)],
                               colors=colors[:len(values)],
                               autopct="%1.1f%%", startangle=90,
                               textprops={"color": LIGHT_TEXT, "fontsize": 9})
        ax.set_aspect("equal")
        return _save_figure(fig, title or "Pie Chart")

    elif chart_type in ("boxplot", "box"):
        data = spec.get("data") or []
        if not data:
            raise DiagramError("boxplot needs 'data' array of arrays")
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(ACCENT)
            patch.set_alpha(0.7)
        for whisker in bp["whiskers"]:
            whisker.set_color(DARK_TICK)
        for cap in bp["caps"]:
            cap.set_color(DARK_TICK)
        for median in bp["medians"]:
            median.set_color(LIGHT_TEXT)
        labels = spec.get("labels") or [f"G{i+1}" for i in range(len(data))]
        ax.set_xticklabels(labels[:len(data)])
        return _save_figure(fig, title or "Box Plot")

    elif chart_type == "scatter":
        xs = spec.get("x") or spec.get("values") or []
        ys = spec.get("y") or []
        if not xs:
            raise DiagramError("scatter needs 'x' and 'y' arrays")
        if not ys:
            ys = xs
            xs = list(range(len(ys)))
        ax.scatter(xs, ys, c=ACCENT, s=40, alpha=0.8, edgecolors="none")
        return _save_figure(fig, title or "Scatter Plot")

    elif chart_type in ("histogram", "hist"):
        data = spec.get("data") or spec.get("values") or []
        if not data:
            raise DiagramError("histogram needs 'data' array")
        bins = int(spec.get("bins", 10))
        bins = max(1, min(100, bins))
        ax.hist(data, bins=bins, color=ACCENT, alpha=0.8, edgecolor=DARK_BG)
        return _save_figure(fig, title or "Histogram")

    elif chart_type == "bar":
        labels = spec.get("labels") or []
        values = spec.get("values") or []
        if not values:
            raise DiagramError("bar chart needs 'values' array")
        if not labels:
            labels = [str(i) for i in range(len(values))]
        x_pos = range(len(values))
        colors = [ACCENT, ACCENT_BLUE, ACCENT_GREEN, ACCENT_YELLOW,
                  "#f472b6", "#a78bfa", "#fb923c", "#2dd4bf"]
        bar_colors = [colors[i % len(colors)] for i in range(len(values))]
        ax.bar(x_pos, values, color=bar_colors, alpha=0.85, edgecolor=DARK_BG)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels[:len(values)], fontsize=8)
        return _save_figure(fig, title or "Bar Chart")

    return _save_figure(fig, title)


# ═══════════════════════════════════════════════════════════════════════
# 3. DOT — Graphviz DOT rendered via networkx + matplotlib
# ═══════════════════════════════════════════════════════════════════════

def _parse_dot(dot_src: str) -> tuple[list, list, dict]:
    """Minimal DOT parser for LLM-generated digraph/graph.

    Returns (nodes, edges, attrs) where:
      nodes = [{"id": "A", "label": "A"}, ...]
      edges = [{"source": "A", "target": "B"}, ...]
      attrs = {"rankdir": "TB", ...}
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    attrs: dict[str, str] = {}

    src = re.sub(r"/\*.*?\*/", "", dot_src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)

    rankdir_m = re.search(r'rankdir\s*=\s*["\']?(\w+)["\']?', src)
    if rankdir_m:
        attrs["rankdir"] = rankdir_m.group(1).upper()

    # Node definitions:  A [label="Foo", shape=box];
    for m in re.finditer(
        r'(\w+)\s*\[([^\]]*)\]',
        re.sub(r'->|--', '  ', src)
    ):
        nid = m.group(1)
        attrs_str = m.group(2)
        label_m = re.search(r'label\s*=\s*["\']([^"\']*)["\']', attrs_str)
        label = label_m.group(1) if label_m else nid
        if not any(n["id"] == nid for n in nodes):
            nodes.append({"id": nid, "label": label})

    # Edges: A -> B;  or  A -- B;
    for m in re.finditer(r'(\w+)\s*(->|--)\s*(\w+)', src):
        src_n, dst_n = m.group(1), m.group(3)
        edges.append({"source": src_n, "target": dst_n})
        for nid in (src_n, dst_n):
            if not any(n["id"] == nid for n in nodes):
                nodes.append({"id": nid, "label": nid})

    return nodes, edges, attrs


def render_dot(spec: dict) -> RenderedFigure:
    dot_src = (spec.get("graph") or spec.get("dot") or "").strip()
    if not dot_src:
        raise DiagramError("DOT spec needs 'graph' field with DOT source")

    nodes, edges, attrs = _parse_dot(dot_src)
    title = (spec.get("title") or "").strip()[:160]

    if not nodes:
        raise DiagramError("no nodes found in DOT source")

    import networkx as nx

    is_directed = "digraph" in dot_src.lower()[:30] or "->" in dot_src
    G = nx.DiGraph() if is_directed else nx.Graph()
    for n in nodes:
        G.add_node(n["id"], label=n.get("label", n["id"]))
    for e in edges:
        G.add_edge(e["source"], e["target"])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    rankdir = attrs.get("rankdir", "TB")
    if rankdir == "LR":
        from networkx.drawing.layout import shell_layout
        pos = shell_layout(G)
    elif rankdir == "RL":
        from networkx.drawing.layout import shell_layout
        pos = shell_layout(G)
        pos = {k: (-v[0], v[1]) for k, v in pos.items()}
    elif rankdir == "BT":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
        pos = {k: (v[0], -v[1]) for k, v in pos.items()}
    else:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ACCENT, node_size=1200,
                           alpha=0.9, edgecolors=DARK_SPINE, linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                            font_size=9, font_color=LIGHT_TEXT,
                            font_weight="bold")

    if is_directed:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=DARK_TICK,
                               arrows=True, arrowsize=15, arrowstyle="->",
                               width=1.5, alpha=0.8,
                               connectionstyle="arc3,rad=0.1")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=DARK_TICK,
                               width=1.5, alpha=0.8)

    if title:
        ax.set_title(title, color=LIGHT_TEXT, fontsize=11, pad=8)

    return _save_figure(fig, title or "Graph")


# ═══════════════════════════════════════════════════════════════════════
# 4. CIRCUIT — electrical circuits via SchemDraw
# ═══════════════════════════════════════════════════════════════════════

ELEMENT_MAP = {
    "resistor": "Resistor",
    "resistor_var": "ResistorVar",
    "capacitor": "Capacitor",
    "capacitor_polar": "CapacitorPolar",
    "inductor": "Inductor",
    "battery": "SourceV",
    "source_v": "SourceV",
    "source_i": "SourceI",
    "diode": "Diode",
    "zener": "Zener",
    "led": "LED",
    "ground": "Ground",
    "line": "Line",
    "dot": "Dot",
    "switch": "Switch",
    "amp": "Opamp",
    "fuse": "Fuse",
    "lamp": "Lamp",
    "ammeter": "MeterA",
    "voltmeter": "MeterV",
}


def render_circuit(spec: dict) -> RenderedFigure:
    elements = spec.get("elements") or []
    if not elements:
        raise DiagramError("circuit spec needs 'elements' array")

    title = (spec.get("title") or "").strip()[:160]

    import schemdraw
    import schemdraw.elements as elm

    # SchemDraw uses its own matplotlib figure
    d = schemdraw.Drawing(show=False)

    direction_stack: list[str] = ["right"]

    for item in elements:
        if not isinstance(item, dict):
            continue
        el_type = (item.get("type") or "line").lower()
        class_name = ELEMENT_MAP.get(el_type)
        if not class_name:
            d.add(elm.Line())
            continue

        cls = getattr(elm, class_name, elm.Line)

        # Directional commands
        direction = (item.get("direction") or "").lower()
        if direction in ("right", "left", "up", "down"):
            getattr(d, "push")()
            direction_stack.append(direction)

        if direction == "down" or direction_stack[-1] == "down":
            if hasattr(cls, "down"):
                elem = cls().down()
            else:
                elem = cls()
        elif direction == "up" or direction_stack[-1] == "up":
            if hasattr(cls, "up"):
                elem = cls().up()
            else:
                elem = cls()
        elif direction == "left" or direction_stack[-1] == "left":
            if hasattr(cls, "left"):
                elem = cls().left()
            else:
                elem = cls()
        else:
            elem = cls()

        label = item.get("label", "")
        value = item.get("value", "")
        full_label = f"{label}\n{value}".strip() if value else label
        if full_label:
            elem.label(full_label)

        d.add(elem)

    d.draw()
    fig = plt.gcf()
    fig.patch.set_facecolor(DARK_BG)
    for ax in fig.axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=DARK_TICK)
        ax.xaxis.label.set_color(LIGHT_TEXT)
        ax.yaxis.label.set_color(LIGHT_TEXT)
        if title:
            ax.set_title(title, color=LIGHT_TEXT, fontsize=11)

    return _save_figure(fig, title)


# ═══════════════════════════════════════════════════════════════════════
# 5. GEOM — geometric diagrams (matplotlib.patches)
# ═══════════════════════════════════════════════════════════════════════

def _draw_angle_mark(ax, vertex, ray1, ray2, radius=0.4, color=ACCENT_YELLOW, alpha=0.6):
    """Draw an arc marking the angle between two rays from a vertex."""
    vx, vy = vertex
    angle1 = math.atan2(ray1[1] - vy, ray1[0] - vx)
    angle2 = math.atan2(ray2[1] - vy, ray2[0] - vx)
    if angle1 < 0:
        angle1 += 2 * math.pi
    if angle2 < 0:
        angle2 += 2 * math.pi
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
    arc = mpatches.Arc(vertex, 2 * radius, 2 * radius, angle=0,
                       theta1=math.degrees(angle1), theta2=math.degrees(angle2),
                       color=color, linewidth=1.5, alpha=alpha)
    ax.add_patch(arc)


def render_geometry(spec: dict) -> RenderedFigure:
    geom_type = (spec.get("type") or "triangle").lower()
    title = (spec.get("title") or "").strip()[:160]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    _dark_axes(ax, title, "", "", grid=False)

    ax.set_aspect("equal")

    if geom_type == "triangle":
        vertices = spec.get("vertices") or [[0, 0], [4, 0], [2, 3]]
        labels = spec.get("labels") or ["A", "B", "C"]
        show_angles = bool(spec.get("show_angles", True))
        show_sides = bool(spec.get("show_sides", True))
        mark_right = bool(spec.get("right_angle"))

        pts = np.array(vertices, dtype=float)
        if pts.shape != (3, 2):
            raise DiagramError("triangle needs exactly 3 vertices")

        # Draw filled triangle
        tri = plt.Polygon(pts, fill=True, facecolor=ACCENT, alpha=0.15,
                          edgecolor=ACCENT, linewidth=2)
        ax.add_patch(tri)

        # Draw vertices
        for i, (x, y) in enumerate(pts):
            ax.plot(x, y, "o", color=ACCENT, markersize=8, zorder=5)
            lab = labels[i] if i < len(labels) else chr(65 + i)
            offset_x = 0.15 * (1 if i != 2 else -0.2)
            offset_y = 0.15 * (1 if i != 1 else -0.4)
            ax.annotate(lab, (x + offset_x, y + offset_y),
                        color=LIGHT_TEXT, fontsize=11, fontweight="bold")

        # Mark right angle
        if mark_right:
            for i in range(3):
                p0 = pts[i]
                p1 = pts[(i + 1) % 3]
                p2 = pts[(i + 2) % 3]
                v1 = p1 - p0
                v2 = p2 - p0
                dot = np.dot(v1, v2)
                if abs(dot) < 0.01:
                    size = 0.4
                    d1 = v1 / np.linalg.norm(v1) * size
                    d2 = v2 / np.linalg.norm(v2) * size
                    corner = p0 + d1 + d2
                    square_pts = [p0 + d1, corner, p0 + d2]
                    sq = plt.Polygon(square_pts, fill=False, edgecolor=ACCENT_YELLOW, linewidth=1.5)
                    ax.add_patch(sq)

        # Mark angles
        if show_angles:
            for i in range(3):
                p0 = pts[i]
                p1 = pts[(i + 1) % 3]
                p2 = pts[(i + 2) % 3]
                v1 = p1 - p0
                v2 = p2 - p0
                dot = np.dot(v1, v2)
                if abs(dot) > 0.01:
                    _draw_angle_mark(ax, p0, p1, p2,
                                     radius=0.35, color=ACCENT_YELLOW, alpha=0.5)

        # Side labels
        if show_sides:
            for i in range(3):
                p0 = pts[i]
                p1 = pts[(i + 1) % 3]
                mid = (p0 + p1) / 2
                length = np.linalg.norm(p1 - p0)
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                norm = np.linalg.norm([dx, dy]) or 1
                nx, ny = -dy / norm * 0.25, dx / norm * 0.25
                ax.annotate(f"{length:.1f}", (mid[0] + nx, mid[1] + ny),
                            color=DARK_TICK, fontsize=8, ha="center")

        # Auto-scale with padding
        all_x = pts[:, 0]
        all_y = pts[:, 1]
        pad_x = max(1.0, (all_x.max() - all_x.min()) * 0.2)
        pad_y = max(1.0, (all_y.max() - all_y.min()) * 0.2)
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    elif geom_type == "circle":
        center = spec.get("center", [0, 0])
        radius = float(spec.get("radius", 3))
        show_axes = bool(spec.get("show_axes", True))

        circ = plt.Circle(center, radius, fill=False, edgecolor=ACCENT, linewidth=2)
        ax.add_patch(circ)
        ax.plot(center[0], center[1], "o", color=ACCENT, markersize=6)

        if show_axes:
            ax.axhline(center[1], color=DARK_SPINE, linewidth=0.5, linestyle="--")
            ax.axvline(center[0], color=DARK_SPINE, linewidth=0.5, linestyle="--")

        margin = radius * 1.4
        ax.set_xlim(center[0] - margin, center[0] + margin)
        ax.set_ylim(center[1] - margin, center[1] + margin)
        ax.set_aspect("equal")

    elif geom_type == "vectors":
        vectors = spec.get("vectors") or []
        if not vectors:
            raise DiagramError("vector geometry needs 'vectors' list of [x,y] or [x,y,dx,dy]")

        origin = spec.get("origin", [0, 0])
        colors_cycle = [ACCENT, ACCENT_BLUE, ACCENT_GREEN, ACCENT_YELLOW]

        max_extent = 1.0
        for i, v in enumerate(vectors):
            if len(v) == 2:
                dx, dy = v[0], v[1]
                ox, oy = origin
            elif len(v) >= 4:
                ox, oy, dx, dy = v[0], v[1], v[2], v[3]
            else:
                continue
            color = colors_cycle[i % len(colors_cycle)]
            ax.arrow(ox, oy, dx, dy, head_width=0.2, head_length=0.3,
                     fc=color, ec=color, linewidth=2, alpha=0.9,
                     length_includes_head=True)
            max_extent = max(max_extent, abs(ox + dx), abs(oy + dy))

        ax.axhline(0, color=DARK_SPINE, linewidth=0.5)
        ax.axvline(0, color=DARK_SPINE, linewidth=0.5)
        margin = max_extent * 0.3 + 0.5
        ax.set_xlim(-max_extent - margin, max_extent + margin)
        ax.set_ylim(-max_extent - margin, max_extent + margin)

    elif geom_type == "ray_optics":
        kind = spec.get("kind", "convex_lens")
        ax.axhline(0, color=DARK_SPINE, linewidth=0.5)
        ax.axvline(0, color=DARK_SPINE, linewidth=1.0)

        if "convex" in kind or "concave" in kind:
            is_convex = "convex" in kind
            h = 3.0
            if is_convex:
                lens = mpatches.Arc((0, 0), 0.6, 2 * h, angle=0, theta1=-90, theta2=90,
                                    color=ACCENT_BLUE, linewidth=2)
            else:
                lens_left = mpatches.Arc((-0.3, 0), 1.2, 2 * h, angle=0, theta1=-90, theta2=90,
                                          color=ACCENT_BLUE, linewidth=2)
                lens_right = mpatches.Arc((0.3, 0), 1.2, 2 * h, angle=0, theta1=90, theta2=270,
                                           color=ACCENT_BLUE, linewidth=2)
                ax.add_patch(lens_left)
                ax.add_patch(lens_right)
            if is_convex:
                ax.add_patch(lens)
            ax.axvline(0, -h, h, color=ACCENT_BLUE, linewidth=0.5, linestyle="--")
            ax.set_xlim(-6, 6)
            ax.set_ylim(-h - 1, h + 1)

        # Draw principal ray
        if spec.get("rays"):
            for ray in spec["rays"]:
                if len(ray) >= 4:
                    ax.plot([ray[0], ray[2]], [ray[1], ray[3]],
                            color=ACCENT_YELLOW, linewidth=1.5, alpha=0.8)

        ax.set_aspect("equal")

    else:
        raise DiagramError(f"unknown geometry type '{geom_type}'")

    ax.tick_params(colors=DARK_TICK, labelsize=8)
    return _save_figure(fig, title)


# ═══════════════════════════════════════════════════════════════════════
# Unified marker scanner + renderer
# ═══════════════════════════════════════════════════════════════════════

_MARKER_TYPES: dict[str, tuple[re.Pattern, callable]] = {
    "PLOT": (re.compile(r"\[\[PLOT:\s*(\{.*?\})\s*\]\]", re.DOTALL), render_plot),
    "CHART": (re.compile(r"\[\[CHART:\s*(\{.*?\})\s*\]\]", re.DOTALL), render_chart),
    "DOT": (re.compile(r"\[\[DOT:\s*(\{.*?\})\s*\]\]", re.DOTALL), render_dot),
    "CIRCUIT": (re.compile(r"\[\[CIRCUIT:\s*(\{.*?\})\s*\]\]", re.DOTALL), render_circuit),
    "GEOM": (re.compile(r"\[\[GEOM:\s*(\{.*?\})\s*\]\]", re.DOTALL), render_geometry),
}


def extract_and_render_all(answer: str) -> tuple[str, list[RenderedFigure]]:
    """Scan answer for all diagram markers, render each, replace with [[FIG:id]].

    Returns (rewritten_answer, figures).
    """
    figures: list[RenderedFigure] = []

    def _repl(maker_type: str, render_fn, raw: str) -> str:
        try:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise DiagramError("spec is not a JSON object")
            fig = render_fn(spec)
            figures.append(fig)
            return f"[[FIG:{fig.id}]]"
        except (json.JSONDecodeError, DiagramError) as e:
            return f"(diagram [{maker_type}]: {e})"
        except Exception as e:
            return f"(diagram [{maker_type}] error: {type(e).__name__})"

    for mtype, (pattern, render_fn) in _MARKER_TYPES.items():
        def _make_repl(mt=mtype, rf=render_fn):
            return lambda m: _repl(mt, rf, m.group(1))
        answer = pattern.sub(_make_repl(), answer)

    return answer, figures
