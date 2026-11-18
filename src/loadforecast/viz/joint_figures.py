"""Figure for the joint (whole-day) forecasts, DESIGN 7.5.

Usage:
    python -m loadforecast.viz.joint_figures --split test

Writes reports/figures/joint_scenarios.png: scenarios of one day from independent hours and
from the flow, and how often each model's 90% range held the outcome for each hour, for the
day's energy and for the day's peak.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loadforecast.config import PROCESSED_DIR, ROOT, load_config  # noqa: E402
from loadforecast.evaluation.backtest import prepare_table  # noqa: E402
from loadforecast.evaluation.run_joint import SCENARIOS_DIR, load_days  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
# Reference categorical palette, first three slots (valid for all pairs).
COLORS = {"independent": "#2a78d6", "copula": "#eb6834", "flow": "#1baf7a"}
LABELS = {"independent": "Independent hours", "copula": "Gaussian copula", "flow": "Spline flow"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def coverage(y: np.ndarray, x: np.ndarray) -> dict:
    """Share of outcomes inside the central 90% range: hours, daily energy, daily peak."""

    def inside(obs, sims):
        lo, hi = np.quantile(sims, [0.05, 0.95], axis=1)
        return float(((obs >= lo) & (obs <= hi)).mean())

    return {
        "Each hour": inside(y, x),
        "Daily energy": inside(y.sum(1), x.sum(2)),
        "Daily peak": inside(y.max(1), x.max(2)),
    }


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    taus = load_config("backtest")["quantiles"]
    table = prepare_table(pd.read_parquet(PROCESSED_DIR / "hourly.parquet"))
    days = load_days(args.split, "rnn", "forecast", table, taus)

    sims = {}
    for name in COLORS:
        z = np.load(SCENARIOS_DIR / args.split / f"{name}.npz")
        sims[name] = (pd.DatetimeIndex(z["dates"]), z["samples"])
    dates = sims["flow"][0]
    sel = days.take(days.dates.isin(dates))
    ok = sel.complete
    y = sel.y[ok]
    rows = {n: np.flatnonzero(d.isin(sel.dates[ok])) for n, (d, _) in sims.items()}

    # The example day: the largest error in daily energy, so the point the panels make is the
    # one the scores measure, chosen without looking at the scenarios.
    energy_error = np.abs(y.sum(1) - sel.point[ok].sum(1))
    k = int(np.argmax(energy_error))
    day = sel.dates[ok][k]

    fig = plt.figure(figsize=(13, 4.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)
    hours = np.arange(24)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, ["independent", "flow"], strict=True):
        x = sims[name][1][rows[name][k]]
        for j in rng.choice(len(x), 30, replace=False):
            ax.plot(hours, x[j] / 1e3, color=COLORS[name], lw=0.8, alpha=0.45)
        ax.plot(hours, y[k] / 1e3, color=INK, lw=2, label="Actual demand")
        ax.set_title(f"{LABELS[name]}: 30 scenarios", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Hour", color=INK_2, fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])
        style(ax)
    axes[0].set_ylabel("GW", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.text(
        0.125,
        0.99,
        f"{day:%A %-d %B %Y}, the day with the largest error in daily energy",
        fontsize=9,
        color=INK_2,
        va="top",
    )

    ax = fig.add_subplot(grid[0, 2])
    cov = {n: coverage(y, sims[n][1][rows[n]]) for n in COLORS}
    groups = list(next(iter(cov.values())))
    width = 0.26
    for i, name in enumerate(COLORS):
        values = [100 * cov[name][g] for g in groups]
        pos = np.arange(len(groups)) + (i - 1) * (width + 0.02)
        ax.bar(pos, values, width, color=COLORS[name], label=LABELS[name], zorder=2)
        for p, v in zip(pos, values, strict=True):
            ax.text(
                p,
                v + 1.5,
                f"{v:.0f}",
                ha="center",
                fontsize=8,
                color=INK_2,
                bbox={"fc": "white", "ec": "none", "pad": 0.6},
                zorder=4,
            )
    ax.axhline(90, color=INK, lw=1, ls="--", zorder=3)
    ax.text(
        1.01,
        90,
        "90%\ntarget",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        color=INK,
        va="center",
    )
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of outcomes inside the 90% range", color=INK_2, fontsize=9)
    ax.set_title("Are the whole-day ranges honest?", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    style(ax)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "joint_scenarios.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} ({day.date()}; energy error {energy_error[k] / 1e3:.0f} GWh)")


if __name__ == "__main__":
    main()
