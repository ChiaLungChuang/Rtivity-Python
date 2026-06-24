"""
Reusable plot-generating functions (all return plotly figures).

Each function accepts tidy DataFrames from the analysis layer and
returns a plotly Figure that can be embedded in the Shiny UI via
ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette shared across all plots
_PALETTE = px.colors.qualitative.Set2


# ── box / strip plot ──────────────────────────────────────────────────────────

def box_plot(
    df: pd.DataFrame,
    title: str = "",
    ylab: str = "Value",
    points: bool = True,
) -> go.Figure:
    """
    Box plot with optional individual points, one color per condition label.
    Input: standard x/y/labels DataFrame from any analysis summary function.
    """
    fig = px.box(
        df.sort_values(["x", "labels"]),
        x="x", y="y", color="labels",
        points="all" if points else False,
        title=title,
        labels={"x": "", "y": ylab, "labels": "Condition"},
        color_discrete_sequence=_PALETTE,
        category_orders={"labels": sorted(df["labels"].unique())},
    )
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Condition",
        font=dict(size=13),
        title_font_size=15,
        margin=dict(t=50, b=40),
    )
    return fig


# ── Kaplan-Meier survival curves ──────────────────────────────────────────────

def km_plot(km_df: pd.DataFrame) -> go.Figure:
    """
    Kaplan-Meier survival curves with shaded 95% CI bands.
    Input: output of kaplan_meier_curves().
    """
    fig = go.Figure()
    labels = km_df["labels"].unique()
    colors = _PALETTE[:len(labels)]

    for label, color in zip(labels, colors):
        grp = km_df[km_df["labels"] == label].sort_values("time")

        # CI shaded band
        x_band = list(grp["time"]) + list(grp["time"])[::-1]
        y_band = list(grp["ci_upper"]) + list(grp["ci_lower"])[::-1]
        r, g, b = _hex_to_rgb(color)
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Survival step line
        fig.add_trace(go.Scatter(
            x=grp["time"], y=grp["survival"],
            mode="lines", name=str(label),
            line=dict(color=color, width=2, shape="hv"),
        ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Days",
        yaxis_title="Survival probability",
        yaxis=dict(range=[0, 1.05]),
        legend_title_text="Condition",
        font=dict(size=13),
    )
    return fig


# ── periodogram ───────────────────────────────────────────────────────────────

def periodogram_plot(
    pgram_df: pd.DataFrame,
    value_col: str = "Q",
    highlight_24h: bool = True,
) -> go.Figure:
    """
    Periodogram: one semi-transparent line per animal, colored by condition.
    Input: output of periodogram_by_animal().
    """
    labels = pgram_df["labels"].unique() if "labels" in pgram_df.columns else ["all"]
    label_color = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(labels)}

    fig = go.Figure()

    for (aid, lbl), grp in pgram_df.groupby(["id", "labels"], sort=False):
        color = label_color.get(lbl, "#888888")
        r, g, b = _hex_to_rgb(color)
        fig.add_trace(go.Scatter(
            x=grp["period_h"], y=grp[value_col],
            mode="lines",
            line=dict(color=f"rgba({r},{g},{b},0.4)", width=1),
            name=str(lbl),
            legendgroup=str(lbl),
            showlegend=(aid == pgram_df[pgram_df["labels"] == lbl]["id"].iloc[0]),
        ))

    if highlight_24h:
        fig.add_vline(x=24.0, line_dash="dash", line_color="gray",
                      annotation_text="24 h", annotation_position="top right")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Period (h)",
        yaxis_title=value_col,
        title="Periodogram",
        legend_title_text="Condition",
        font=dict(size=13),
    )
    return fig


# ── actogram ──────────────────────────────────────────────────────────────────

def actogram_plot(
    df: pd.DataFrame,
    animal_id: str,
    bin_min: int = 30,
) -> go.Figure:
    """
    Double-plotted actogram for a single animal.

    Each row represents one day.  The left half (0–24 h) shows that day's
    data; the right half (24–48 h) repeats the next day's data — standard
    chronobiology convention.
    """
    grp = df[df["id"] == animal_id].copy()
    if grp.empty:
        return go.Figure()

    bin_s = bin_min * 60.0
    day_s = 24.0 * 3600.0
    bins_per_day = round(day_s / bin_s)

    # Bin activity counts
    grp["_bin"] = (grp["t"] // bin_s).astype(int)
    binned = grp.groupby("_bin")["activity"].sum()

    max_day = int(grp["t"].max() // day_s) + 1
    mat = np.zeros((max_day, 2 * bins_per_day))

    for tbin, act in binned.items():
        day_idx   = tbin // bins_per_day
        within    = tbin % bins_per_day
        # Left half: current day
        if 0 <= day_idx < max_day:
            mat[day_idx, within] += act
        # Right half: same data appears as next day's left half
        # ↔ appears in row (day_idx-1), col (bins_per_day + within)
        if 0 <= day_idx - 1 < max_day:
            mat[day_idx - 1, bins_per_day + within] += act

    zt_hours = np.arange(2 * bins_per_day) * bin_min / 60.0
    day_labels = [f"Day {d}" for d in range(max_day)]

    fig = go.Figure(go.Heatmap(
        z=mat,
        x=zt_hours,
        y=day_labels,
        colorscale="Blues",
        showscale=False,
        hovertemplate="ZT %{x:.1f} h<br>%{y}<br>Activity: %{z}<extra></extra>",
    ))

    # Mark lights-off at ZT12 and ZT36
    for zt in (12.0, 36.0):
        fig.add_vline(x=zt, line_color="rgba(255,200,0,0.6)", line_width=1.5)

    fig.update_yaxes(autorange="reversed", title="Day")
    fig.update_xaxes(title="ZT (h)", tickvals=list(range(0, 49, 6)))
    fig.update_layout(
        title=f"Actogram — {animal_id}",
        template="plotly_white",
        font=dict(size=12),
        height=max(300, max_day * 28 + 80),
    )
    return fig


# ── average daily activity profile ────────────────────────────────────────────

def daily_profile_plot(
    df: pd.DataFrame,
    bin_min: int = 30,
) -> go.Figure:
    """
    Average activity profile over one 24-h day, one line per condition group.
    """
    bin_s = bin_min * 60.0
    day_s = 24.0 * 3600.0
    bins_per_day = round(day_s / bin_s)

    df = df.copy()
    df["_bin"] = ((df["t"] % day_s) // bin_s).astype(int)
    zt_h = np.arange(bins_per_day) * bin_min / 60.0

    fig = go.Figure()
    for i, (label, grp) in enumerate(df.groupby("labels", sort=False)):
        color = _PALETTE[i % len(_PALETTE)]
        r, g, b = _hex_to_rgb(color)

        profile = (
            grp.groupby(["id", "_bin"])["activity"].mean()
            .groupby("_bin").agg(["mean", "sem"])
            .reindex(range(bins_per_day))
            .fillna(0)
        )

        # SEM band
        fig.add_trace(go.Scatter(
            x=list(zt_h) + list(zt_h[::-1]),
            y=list(profile["mean"] + profile["sem"]) +
              list((profile["mean"] - profile["sem"])[::-1]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=zt_h, y=profile["mean"],
            mode="lines", name=str(label),
            line=dict(color=color, width=2),
        ))

    fig.add_vrect(x0=12, x1=24, fillcolor="rgba(0,0,0,0.07)",
                  line_width=0, annotation_text="Dark")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="ZT (h)", yaxis_title="Mean activity (counts/min)",
        legend_title_text="Condition",
        title="Average Daily Activity Profile",
        font=dict(size=13),
        xaxis=dict(tickvals=list(range(0, 25, 6))),
    )
    return fig


# ── helpers ───────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' or 'rgb(r,g,b)' to (r, g, b) ints."""
    h = hex_color.strip()
    if h.startswith("rgb"):
        parts = h.replace("rgb(", "").replace(")", "").split(",")
        return tuple(int(x.strip()) for x in parts)
    h = h.lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 100, 100, 100
