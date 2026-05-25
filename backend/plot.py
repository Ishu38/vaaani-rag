"""Safe server-side matplotlib rendering for LLM-requested plots.

The LLM emits inline markers like:

    [[PLOT:{"expr": "sin(x)", "x_min": -3.14159, "x_max": 3.14159,
            "title": "y = sin(x)"}]]

`render_spec` parses one such spec, compiles the expression via SymPy (no
`eval`, no `exec`), evaluates with NumPy, and writes a PNG. The caller is
responsible for embedding the PNG URL in the response.

Why SymPy and not asteval / eval-with-restricted-globals: SymPy guarantees the
parsed AST contains only mathematical operations from a whitelisted symbol set.
A bad expression raises `SympifyError` before any numerical evaluation runs.
"""
from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

# Headless backend — matplotlib must never try to open a window on the VPS.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Whitelisted SymPy symbols available inside `expr`. Anything else triggers
# a SympifyError. `pi` and `E` are constants; everything else is a function.
_ALLOWED_LOCALS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs,
    "pi": sp.pi, "E": sp.E, "e": sp.E,
}

# Hard caps so a malicious / clumsy spec can't blow up the VPS.
MAX_POINTS = 2000
MIN_RANGE_SPAN = 1e-6
MAX_RANGE_SPAN = 1e4
PLOT_DIR_NAME = "figures"


@dataclass
class RenderedFigure:
    """Result of one render_spec call — what the route handler embeds in the response."""
    id: str
    path: Path
    url: str
    caption: str
    expr: str


class PlotSpecError(ValueError):
    """Raised when a spec is malformed or unsafe — caller turns it into a no-op."""


def _safe_lambdify(expr_str: str) -> tuple[sp.Expr, Any]:
    """Parse `expr_str` to a SymPy expression of a single variable `x`, then
    lambdify it to a NumPy callable. Raises `PlotSpecError` on anything fishy.
    """
    if not expr_str or len(expr_str) > 200:
        raise PlotSpecError("expr is empty or too long")
    try:
        # `sympify` with explicit locals refuses unknown symbols.
        x = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={**_ALLOWED_LOCALS, "x": x})
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise PlotSpecError(f"could not parse expression: {e}")
    # Reject anything that depends on a symbol other than `x`.
    free = expr.free_symbols
    if free and free != {x}:
        raise PlotSpecError(f"expression must depend only on x; saw {sorted(map(str, free))}")
    return expr, sp.lambdify(x, expr, modules=["numpy"])


def _coerce_range(spec: dict) -> tuple[float, float]:
    """Validate x_min < x_max and the span is sane."""
    try:
        x_min = float(spec.get("x_min", -10.0))
        x_max = float(spec.get("x_max", 10.0))
    except (TypeError, ValueError) as e:
        raise PlotSpecError(f"x_min/x_max must be numbers: {e}")
    if not (x_min < x_max):
        raise PlotSpecError("x_min must be strictly less than x_max")
    span = x_max - x_min
    if span < MIN_RANGE_SPAN or span > MAX_RANGE_SPAN:
        raise PlotSpecError(f"range span {span} is out of bounds")
    return x_min, x_max


def render_spec(spec: dict, *, out_dir: Path) -> RenderedFigure:
    """Render one plot spec to disk; return the embeddable metadata.

    Spec fields recognised:
      expr        (str)   — function of x, e.g. "sin(x)" or "x**2 - 4"
      x_min       (float) — default -10
      x_max       (float) — default 10
      title       (str)   — optional; defaults to "y = <expr>"
      x_label     (str)   — optional axis label
      y_label     (str)   — optional axis label
      mark_zero   (bool)  — whether to draw x=0 and y=0 axis lines (default true)
    """
    expr_str = (spec.get("expr") or "").strip()
    expr, fn = _safe_lambdify(expr_str)
    x_min, x_max = _coerce_range(spec)
    title = (spec.get("title") or f"y = {sp.pretty(expr, use_unicode=False)}")[:160]
    x_label = (spec.get("x_label") or "x")[:60]
    y_label = (spec.get("y_label") or f"f(x)")[:60]
    mark_zero = bool(spec.get("mark_zero", True))

    # Evaluate. We catch ZeroDivisionError / overflow at the numpy stage and
    # mask non-finite values so the plot stays smooth.
    xs = np.linspace(x_min, x_max, MAX_POINTS)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ys = fn(xs)
    ys = np.where(np.isfinite(ys), ys, np.nan)

    # Dark theme matching the chat UI (var(--bg) #0a0a0c, var(--text) #ececf1).
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    fig.patch.set_facecolor("#131318")
    ax.set_facecolor("#131318")
    for spine in ax.spines.values():
        spine.set_color("#3a3a48")
    ax.tick_params(colors="#9a9aa8", labelsize=9)
    ax.xaxis.label.set_color("#ececf1")
    ax.yaxis.label.set_color("#ececf1")
    ax.title.set_color("#ececf1")
    ax.grid(True, color="#23232c", linewidth=0.5)

    if mark_zero:
        ax.axhline(0, color="#3a3a48", linewidth=0.8)
        ax.axvline(0, color="#3a3a48", linewidth=0.8)

    ax.plot(xs, ys, color="#e63b66", linewidth=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, pad=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_id = uuid.uuid4().hex[:12]
    out_path = out_dir / f"{fig_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    return RenderedFigure(
        id=fig_id,
        path=out_path,
        url=f"/figures/{fig_id}.png",
        caption=title,
        expr=expr_str,
    )


# Matches [[PLOT:{...json...}]]. Lazy match on JSON body. Tolerates whitespace.
_PLOT_MARKER = re.compile(r"\[\[PLOT:\s*(\{.*?\})\s*\]\]", re.DOTALL)


def extract_and_render(answer: str, *, out_dir: Path) -> tuple[str, list[RenderedFigure]]:
    """Scan `answer` for [[PLOT:...]] markers, render each, replace with [[FIG:id]].

    Returns (rewritten_answer, figures). On any per-marker failure (bad JSON,
    unsafe expression, unknown function), the marker is replaced with a
    short apology line and the failure does not propagate.
    """
    figures: list[RenderedFigure] = []

    def _repl(m: re.Match) -> str:
        raw = m.group(1)
        try:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise PlotSpecError("spec is not a JSON object")
            fig = render_spec(spec, out_dir=out_dir)
            figures.append(fig)
            return f"[[FIG:{fig.id}]]"
        except (json.JSONDecodeError, PlotSpecError) as e:
            return f"(figure could not be rendered: {e})"
        except Exception as e:  # last-resort guard so /chat never 500s on plotting
            return f"(figure error: {type(e).__name__})"

    rewritten = _PLOT_MARKER.sub(_repl, answer)
    return rewritten, figures
