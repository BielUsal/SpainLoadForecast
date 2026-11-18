"""Dashboard: day-ahead forecasts of Spanish electricity demand against what happened.

Run locally with ``make app`` (or ``streamlit run app/streamlit_app.py``). The app reads only
the small bundle in ``app/data/`` (built by ``python -m loadforecast.viz.app_data``), so it
does not need the project package or the raw data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
SPLIT = "test"

# Display name and colour of each model. Colours belong to the model, not to its position in
# a chart, so a model keeps its colour whatever else is selected.
MODELS = {
    "ree": ("REE (grid operator)", "#2a78d6"),
    "lgbm_forecast": ("LightGBM", "#eb6834"),
    "bayes_nuts_forecast": ("Bayesian model", "#1baf7a"),
    "rnn_forecast": ("Recurrent neural network (RNN)", "#c8303c"),
    "climatology": ("Baseline: typical profile", "#eda100"),
    "naive_weekly": ("Baseline: same hour last week", "#e87ba4"),
    "lgbm_oracle": ("LightGBM, perfect weather", "#008300"),
    "bayes_nuts_oracle": ("Bayesian model, perfect weather", "#6250d6"),
    "rnn_oracle": ("RNN, perfect weather", "#8a6a1f"),
}
DEFAULT_MODELS = ["ree", "lgbm_forecast", "rnn_forecast"]
ACTUAL_COLOR = "#52514e"
DAY_KINDS = [
    "Working day",
    "Saturday",
    "Sunday",
    "Holiday",
    "Bridge day",
    "Christmas period",
    "Holy Week",
]
# Whole-day models (the report's figure uses the same colours). They are a separate family
# from the forecasts above: all three share the RNN's hourly ranges.
JOINT = {
    "independent": ("Independent hours", "#2a78d6"),
    "copula": ("Gaussian copula", "#eb6834"),
    "flow": ("Normalizing flow", "#1baf7a"),
}
TEMP_BINS = [-50, 5, 10, 15, 20, 25, 60]
TEMP_LABELS = ["< 5", "5–10", "10–15", "15–20", "20–25", "> 25"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def name(model: str) -> str:
    return MODELS[model][0]


def color(model: str) -> str:
    return MODELS[model][1]


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    actuals = pd.read_parquet(DATA / f"{SPLIT}_actuals.parquet")
    predictions = pd.read_parquet(DATA / f"{SPLIT}_predictions.parquet")
    predictions["model"] = predictions["model"].astype(str)
    scores = pd.read_csv(DATA / f"{SPLIT}_scores.csv", index_col=0)
    meta = json.loads((DATA / f"{SPLIT}_meta.json").read_text())
    actuals["local_time"] = actuals["local_date"] + pd.to_timedelta(actuals["local_hour"], "h")
    return actuals, predictions, scores, meta


@st.cache_data
def scored_hours() -> pd.DataFrame:
    """Every scorable model-hour with its error, joined to the context of the hour."""
    actuals, predictions, _, _ = load()
    context = actuals[["local_date", "local_hour", "is_masked", "demand_mw", "temp_c", "day_kind"]]
    out = predictions.join(context)
    out = out[~out["is_masked"]].dropna(subset=["point", "demand_mw"])
    out["error"] = out["point"] - out["demand_mw"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = 100 * out["abs_error"] / out["demand_mw"]
    return out


def base_layout(fig: go.Figure, y_title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title,
    )
    return fig


# --- tab 1: one day -----------------------------------------------------------------------


def day_tab(actuals: pd.DataFrame, predictions: pd.DataFrame, meta: dict) -> None:
    first, last = actuals["local_date"].min().date(), actuals["local_date"].max().date()
    left, middle, right = st.columns([1, 1, 2])
    day = left.date_input(
        "First day",
        value=min(first + pd.Timedelta(days=74), last),  # mid-January, a winter working day
        min_value=first,
        max_value=last,
    )
    n_days = middle.selectbox("Days to show", [1, 3, 7], index=0)
    chosen = right.multiselect("Forecasts", list(MODELS), default=DEFAULT_MODELS, format_func=name)
    band = st.radio(
        "Uncertainty range",
        ["90%", "50%", "None"],
        horizontal=True,
        help="The range each model expects demand to fall in with that probability. "
        "With several forecasts selected the ranges overlap; pick one forecast to read them.",
    )

    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=n_days - 1)
    window = actuals[(actuals["local_date"] >= start) & (actuals["local_date"] <= end)]
    if window.empty:
        st.info("No data for this day.")
        return

    fig = go.Figure()
    limits = {"90%": ("q0.05", "q0.95"), "50%": ("q0.25", "q0.75")}.get(band)
    for model in chosen:
        pred = predictions[predictions["model"] == model].reindex(window.index)
        if limits:
            low, high = limits
            fig.add_trace(
                go.Scatter(
                    x=window["local_time"],
                    y=pred[high],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=window["local_time"],
                    y=pred[low],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba(color(model), 0.18),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    for model in chosen:
        pred = predictions[predictions["model"] == model].reindex(window.index)
        fig.add_trace(
            go.Scatter(
                x=window["local_time"],
                y=pred["point"],
                mode="lines",
                name=name(model),
                line=dict(color=color(model), width=2),
                hovertemplate="%{y:,.0f} MW",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=window["local_time"],
            y=window["demand_mw"],
            mode="lines",
            name="Actual demand",
            line=dict(color=ACTUAL_COLOR, width=3),
            hovertemplate="%{y:,.0f} MW",
        )
    )
    base_layout(fig, "Demand (MW)", height=440)
    fig.update_xaxes(hoverformat="%a %d %b %Y, %H:%M")
    st.plotly_chart(fig, width="stretch")

    if window["is_masked"].any():
        st.warning(
            "This period includes a day that is excluded from every score because the "
            "measured values are not ordinary demand: the Iberian blackout of 28–29 April "
            "2025, or 11 June 2026, when demand fell far below every forecast."
        )

    kinds = ", ".join(window.groupby("local_date")["day_kind"].first().astype(str).unique())
    st.caption(
        f"{kinds} · mean temperature {window['temp_c'].mean():.1f} °C · forecasts issued at "
        f"{meta['issue_time_local']} the day before each target day"
    )

    scorable = window[~window["is_masked"]]
    if chosen and not scorable.empty:
        rows = []
        for model in chosen:
            pred = predictions[predictions["model"] == model].reindex(scorable.index)
            err = (pred["point"] - scorable["demand_mw"]).abs()
            inside = (scorable["demand_mw"] >= pred["q0.05"]) & (
                scorable["demand_mw"] <= pred["q0.95"]
            )
            rows.append(
                {
                    "Forecast": name(model),
                    "Mean error (MW)": err.mean(),
                    "Mean error (%)": (100 * err / scorable["demand_mw"]).mean(),
                    "Largest error (MW)": err.max(),
                    "Hours inside the 90% range": f"{int(inside.sum())} of {len(scorable)}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("Forecast"),
            column_config={
                "Mean error (MW)": st.column_config.NumberColumn(format="%.0f"),
                "Mean error (%)": st.column_config.NumberColumn(format="%.2f"),
                "Largest error (MW)": st.column_config.NumberColumn(format="%.0f"),
            },
            width="stretch",
        )


# --- tab 2: accuracy over the year --------------------------------------------------------


def grouped_chart(table: pd.DataFrame, x_title: str, y_title: str, kind: str) -> go.Figure:
    """One line or one group of bars per model; ``table`` is (group, model) values."""
    fig = go.Figure()
    for model in table.columns:
        common = dict(x=table.index, y=table[model], name=name(model), hovertemplate="%{y:,.2f}")
        if kind == "line":
            fig.add_trace(
                go.Scatter(
                    mode="lines+markers",
                    marker=dict(size=7),
                    line=dict(color=color(model), width=2),
                    **common,
                )
            )
        else:
            fig.add_trace(go.Bar(marker_color=color(model), **common))
    base_layout(fig, y_title, height=340)
    fig.update_layout(xaxis_title=x_title, bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(rangemode="tozero")
    return fig


def accuracy_tab(scores: pd.DataFrame) -> None:
    hours = scored_hours()
    table = scores.loc[[m for m in MODELS if m in scores.index]]
    shown = (
        pd.DataFrame(
            {
                "Forecast": [name(m) for m in table.index],
                "Mean error (MW)": table["MAE"].to_numpy(),
                "Mean error (%)": table["MAPE"].to_numpy(),
                "Probabilistic score, CRPS (MW)": table["CRPS"].to_numpy(),
                "Coverage of the 90% range (%)": table["cov90"].to_numpy(),
                "Width of the 90% range (MW)": table["width90"].to_numpy(),
            }
        )
        .set_index("Forecast")
        .sort_values("Mean error (MW)")
    )
    st.subheader("The whole year in one table")
    st.dataframe(
        shown,
        column_config={
            "Mean error (MW)": st.column_config.NumberColumn(format="%.0f"),
            "Mean error (%)": st.column_config.NumberColumn(format="%.2f"),
            "Probabilistic score, CRPS (MW)": st.column_config.NumberColumn(format="%.0f"),
            "Coverage of the 90% range (%)": st.column_config.NumberColumn(format="%.1f"),
            "Width of the 90% range (MW)": st.column_config.NumberColumn(format="%.0f"),
        },
        width="stretch",
    )
    st.caption(
        "Lower is better for the errors and for CRPS, which scores the whole range and not "
        "only its centre. A 90% range is honest if it contains the outcome about 90% of the "
        'time. "Perfect weather" variants are given the weather that was later observed: '
        "they show how much the weather forecast costs, and are not real forecasts."
    )

    st.subheader("Where the errors are")
    left, right = st.columns([3, 1])
    chosen = left.multiselect(
        "Forecasts", list(MODELS), default=DEFAULT_MODELS, format_func=name, key="accuracy_models"
    )
    metric = right.radio("Error measure", ["%", "MW"], horizontal=True)
    column, y_title = (
        ("ape", "Mean error (%)") if metric == "%" else ("abs_error", "Mean error (MW)")
    )
    if not chosen:
        st.info("Select at least one forecast.")
        return
    part = hours[hours["model"].isin(chosen)]

    def by(group: pd.Series) -> pd.DataFrame:
        return part.groupby([group, part["model"]], observed=True)[column].mean().unstack()[chosen]

    first, second = st.columns(2)
    first.markdown("**By hour of the day**")
    first.plotly_chart(
        grouped_chart(by(part["local_hour"]), "Hour (local time)", y_title, "line"),
        width="stretch",
    )
    monthly = by(part["local_date"].dt.month)
    monthly.index = [MONTHS[m - 1] for m in monthly.index]
    second.markdown("**By month**")
    second.plotly_chart(grouped_chart(monthly, "", y_title, "line"), width="stretch")

    third, fourth = st.columns(2)
    kinds = by(part["day_kind"].astype(str)).reindex(DAY_KINDS).dropna(how="all")
    third.markdown("**By kind of day**")
    third.plotly_chart(grouped_chart(kinds, "", y_title, "bar"), width="stretch")
    temperature = by(pd.cut(part["temp_c"], TEMP_BINS, labels=TEMP_LABELS))
    temperature.index = temperature.index.astype(str)
    fourth.markdown("**By temperature**")
    fourth.plotly_chart(
        grouped_chart(temperature, "National mean temperature (°C)", y_title, "bar"),
        width="stretch",
    )
    days = hours.drop_duplicates(["local_date"]).groupby("day_kind", observed=True).size()
    st.caption(
        "Days in the year by kind: "
        + ", ".join(f"{k} {int(days.get(k, 0))}" for k in DAY_KINDS)
        + ". Groups with few days are noisy."
    )


# --- tab 3: calibration -------------------------------------------------------------------


def calibration_tab(meta: dict) -> None:
    hours = scored_hours()
    chosen = st.multiselect(
        "Forecasts",
        list(MODELS),
        default=DEFAULT_MODELS,
        format_func=name,
        key="calibration_models",
    )
    if not chosen:
        st.info("Select at least one forecast.")
        return
    taus = meta["quantiles"]
    left, right = st.columns(2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfectly honest",
            line=dict(color=ACTUAL_COLOR, width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    for model in chosen:
        part = hours[hours["model"] == model].dropna(subset=[f"q{t}" for t in taus])
        observed = [100 * (part["demand_mw"] <= part[f"q{t}"]).mean() for t in taus]
        fig.add_trace(
            go.Scatter(
                x=[100 * t for t in taus],
                y=observed,
                mode="lines+markers",
                name=name(model),
                marker=dict(size=8),
                line=dict(color=color(model), width=2),
                hovertemplate="%{y:.1f}% of outcomes below",
            )
        )
    base_layout(fig, "Share of outcomes below the level (%)")
    fig.update_layout(xaxis_title="Level the model announced (%)")
    left.markdown("**Do the announced probabilities come true?**")
    left.plotly_chart(fig, width="stretch")
    left.caption(
        'Each model says, for example, "demand will be below this value with 25% '
        'probability". Over the year that should be true 25% of the time. Points on the '
        "dotted line are honest; a curve below it means the model's values run too low."
    )

    fig = go.Figure()
    fig.add_hline(y=90, line=dict(color=ACTUAL_COLOR, width=1, dash="dot"))
    for model in chosen:
        part = hours[hours["model"] == model].dropna(subset=["q0.05", "q0.95"])
        inside = (part["demand_mw"] >= part["q0.05"]) & (part["demand_mw"] <= part["q0.95"])
        monthly = 100 * inside.groupby(part["local_date"].dt.to_period("M")).mean()
        fig.add_trace(
            go.Scatter(
                x=monthly.index.to_timestamp(),
                y=monthly.to_numpy(),
                mode="lines+markers",
                name=name(model),
                marker=dict(size=7),
                line=dict(color=color(model), width=2),
                hovertemplate="%{y:.1f}%",
            )
        )
    base_layout(fig, "Outcomes inside the 90% range (%)")
    fig.update_xaxes(hoverformat="%b %Y", dtick="M2", tickformat="%b %Y")
    right.markdown("**Coverage of the 90% range, month by month**")
    right.plotly_chart(fig, width="stretch")
    right.caption(
        "The dotted line is the target. Below it the ranges were too narrow that month; "
        "above it, wider than they needed to be."
    )


# --- tab 4: whole-day scenarios ----------------------------------------------------------


@st.cache_data
def load_scenarios() -> dict | None:
    path = DATA / f"{SPLIT}_scenarios.npz"
    if not path.exists():
        return None
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["dates"] = pd.DatetimeIndex(out["dates"])
    out["models"] = [str(m) for m in out["models"]]
    out["scores"] = pd.read_csv(DATA / f"{SPLIT}_joint_scores.csv", index_col=0)
    return out


def day_matrix(actuals: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Demand as (days, 24) for ``dates``, and which days count in the scores."""
    wide = actuals.pivot_table(index="local_date", columns="local_hour", values="demand_mw")
    masked = actuals.groupby("local_date")["is_masked"].any()
    wide = wide.reindex(dates)
    ok = wide.notna().all(axis=1).to_numpy() & ~masked.reindex(dates, fill_value=True).to_numpy()
    return wide.to_numpy(), ok


def scenarios_tab(actuals: pd.DataFrame, predictions: pd.DataFrame) -> None:
    sc = load_scenarios()
    if sc is None:
        st.info("No scenarios in this build of the app.")
        return
    models = [m for m in JOINT if m in sc["models"]]
    index = {m: sc["models"].index(m) for m in models}
    st.markdown(
        "The other tabs judge each hour on its own. But some questions are about the whole "
        "day: how much energy will Spain use tomorrow, or how high will the evening peak be? "
        "For those one needs **scenarios**: complete 24-hour paths that could happen. All three "
        "models below start from the RNN's forecast and its range for each hour, so they agree "
        "hour by hour. They differ in how the hours move together:"
    )
    st.markdown(
        "- **Independent hours** draws each hour on its own. Errors in one hour say nothing "
        "about the next, which is not how demand behaves.\n"
        "- **Gaussian copula** learns from the past year how the RNN's errors in different hours "
        "go together, and draws them jointly.\n"
        "- **Normalizing flow** is a small neural network that learns that joint behaviour "
        "too, and can let it change with the season, the kind of day and the weather."
    )

    first, last = sc["dates"].min().date(), sc["dates"].max().date()
    left, right = st.columns([1, 2])
    day = left.date_input(
        "Day",
        value=min(first + pd.Timedelta(days=74), last),
        min_value=first,
        max_value=last,
        key="scenario_day",
    )
    model = right.radio(
        "Scenarios from",
        models,
        index=models.index("flow") if "flow" in models else 0,
        format_func=lambda m: JOINT[m][0],
        horizontal=True,
    )
    day = pd.Timestamp(day)
    if day not in sc["dates"]:
        st.info("No scenarios for this day: the clock changes on it, so it does not have 24 hours.")
        return
    k = sc["dates"].get_loc(day)
    y, ok = day_matrix(actuals, sc["dates"][[k]])
    y, ok = y[0], bool(ok[0])
    hours = np.arange(24)

    plot, side = st.columns([2, 1])
    fig = go.Figure()
    x = sc["samples"][index[model], k].astype(float)
    for j in range(min(30, len(x))):
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=x[j],
                mode="lines",
                line=dict(color=rgba(JOINT[model][1], 0.4), width=1),
                hoverinfo="skip",
                showlegend=j == 0,
                name=f"{JOINT[model][0]}: 30 scenarios",
            )
        )
    rows = actuals[actuals["local_date"] == day].index
    rnn = predictions[predictions["model"] == "rnn_forecast"].reindex(rows)
    fig.add_trace(
        go.Scatter(
            x=actuals.loc[rows, "local_hour"],
            y=rnn["point"],
            mode="lines",
            name="RNN forecast",
            line=dict(color=color("rnn_forecast"), width=2, dash="dash"),
            hovertemplate="%{y:,.0f} MW",
        )
    )
    if np.isfinite(y).all():
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=y,
                mode="lines",
                name="Actual demand",
                line=dict(color=ACTUAL_COLOR, width=3),
                hovertemplate="%{y:,.0f} MW",
            )
        )
    base_layout(fig, "Demand (MW)", height=420)
    fig.update_layout(xaxis_title="Hour (local time)")
    plot.markdown(f"**{day:%A %-d %B %Y}**")
    plot.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    actual_energy = float(np.sum(y)) / 1e3 if np.isfinite(y).all() else None
    for m in models:
        lo, mid, hi = sc["energy_q"][index[m], k]
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[JOINT[m][0]] * 2,
                mode="lines",
                line=dict(color=JOINT[m][1], width=10),
                showlegend=False,
                hovertemplate=f"90% range {lo:,.0f} to {hi:,.0f} GWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mid],
                y=[JOINT[m][0]],
                mode="markers",
                marker=dict(color="white", size=9, line=dict(color=JOINT[m][1], width=2)),
                showlegend=False,
                hovertemplate=f"median {mid:,.0f} GWh<extra></extra>",
            )
        )
    if actual_energy is not None:
        fig.add_vline(x=actual_energy, line=dict(color=ACTUAL_COLOR, width=2, dash="dot"))
        fig.add_annotation(
            x=actual_energy,
            y=1.02,
            yref="paper",
            text="actual",
            showarrow=False,
            font=dict(color=ACTUAL_COLOR),
        )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Energy used over the day (GWh)",
        yaxis=dict(autorange="reversed"),
    )
    side.markdown("**The day's total energy: 90% range**")
    side.plotly_chart(fig, width="stretch")
    side.caption(
        "Independent hours lets the errors of different hours cancel out, so its range for "
        "the day's total is far too narrow. The other two know that a day which starts above "
        "the forecast tends to stay above it."
    )
    if not ok:
        st.warning("This day is excluded from the scores (see About).")

    st.subheader("Over the whole year")
    y_all, ok_all = day_matrix(actuals, sc["dates"])
    y_all = y_all[ok_all]
    coverage = {}
    for m in models:
        i = index[m]
        hq, eq, pq = sc["hour_q"][i][ok_all], sc["energy_q"][i][ok_all], sc["peak_q"][i][ok_all]
        energy, peak = y_all.sum(1) / 1e3, y_all.max(1)
        coverage[m] = {
            "Each hour": ((y_all >= hq[:, 0]) & (y_all <= hq[:, 1])).mean(),
            "Daily energy": ((energy >= eq[:, 0]) & (energy <= eq[:, 2])).mean(),
            "Daily peak": ((peak >= pq[:, 0]) & (peak <= pq[:, 2])).mean(),
        }
    left, right = st.columns([1, 1])
    fig = go.Figure()
    fig.add_hline(y=90, line=dict(color=ACTUAL_COLOR, width=1, dash="dot"))
    for m in models:
        values = [100 * v for v in coverage[m].values()]
        fig.add_trace(
            go.Bar(
                x=list(coverage[m]),
                y=values,
                name=JOINT[m][0],
                marker_color=JOINT[m][1],
                text=[f"{v:.0f}" for v in values],
                textposition="outside",
                hovertemplate="%{y:.1f}%",
            )
        )
    base_layout(fig, "Outcomes inside the 90% range (%)", height=360)
    fig.update_layout(bargap=0.25, bargroupgap=0.08, hovermode="closest")
    fig.update_yaxes(range=[0, 105])
    left.markdown("**Are the whole-day ranges honest?**")
    left.plotly_chart(fig, width="stretch")
    left.caption(
        "The dotted line is the target. Each hour on its own is fine for all three; the "
        "day's total and its peak are only right when the hours move together."
    )

    scores = sc["scores"].reindex(models)
    shown = pd.DataFrame(
        {
            "Model": [JOINT[m][0] for m in models],
            "Energy score (MW)": scores["energy"].to_numpy(),
            "Peak, CRPS (MW)": scores["crps_peak"].to_numpy(),
            "Daily energy, CRPS (GWh)": scores["crps_energy_gwh"].to_numpy(),
            "Change vs independent": [
                "" if pd.isna(v) else f"{v:+.0f} ({ci})"
                for v, ci in zip(
                    scores["energy_minus_independent"],
                    scores["energy_minus_independent_ci"],
                    strict=True,
                )
            ],
        }
    ).set_index("Model")
    right.markdown("**Scores over the year**")
    right.dataframe(
        shown,
        column_config={
            "Energy score (MW)": st.column_config.NumberColumn(format="%.0f"),
            "Peak, CRPS (MW)": st.column_config.NumberColumn(format="%.0f"),
            "Daily energy, CRPS (GWh)": st.column_config.NumberColumn(format="%.2f"),
        },
        width="stretch",
    )
    right.caption(
        "Lower is better. The energy score judges whole 24-hour paths at once. The last "
        "column is the change in energy score against independent hours, with a 95% "
        f"interval, over {int(scores['days'].iloc[0])} days. The flow and the copula are "
        "close: the flow is ahead on most scores, but not by more than chance would allow."
    )


# --- page ---------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Spain load forecast", page_icon="⚡", layout="wide")
    actuals, predictions, scores, meta = load()
    st.title("How much electricity will Spain use tomorrow?")
    st.markdown(
        "Day-ahead forecasts of hourly electricity demand in mainland Spain, each with an "
        "uncertainty range, compared with what happened and with the forecast published by "
        "the grid operator, Red Eléctrica (REE)."
    )
    st.info(
        f"These results are for the **test year**, {meta['start']} to {meta['end']}, scored once "
        "after every choice had been fixed on the year before it.",
        icon="ℹ️",
    )
    day, accuracy, calibration, scenarios, about = st.tabs(
        [
            "Explore a day",
            "Accuracy over the year",
            "Are the ranges honest?",
            "Whole-day scenarios",
            "About",
        ]
    )
    with day:
        day_tab(actuals, predictions, meta)
    with accuracy:
        accuracy_tab(scores)
    with calibration:
        calibration_tab(meta)
    with scenarios:
        scenarios_tab(actuals, predictions)
    with about:
        st.markdown(
            f"""
**What is forecast.** Hourly electricity demand of the Spanish mainland, for the 24 hours of
the next day. Every forecast is made at {meta["issue_time_local"]} (Madrid time) the day
before, using only what was known at that moment.

**The forecasts**

- **REE (grid operator):** the day-ahead forecast REE publishes at about 10:50 the day before.
  It is the professional benchmark. REE publishes no uncertainty range; the one shown here is
  built from its own past errors.
- **LightGBM:** gradient-boosted trees on recent demand, the calendar and weather forecasts.
- **Bayesian model:** a statistical model with readable parts: daily and weekly rhythms,
  holidays, heating and cooling as a function of temperature, and the growth of rooftop solar.
- **Recurrent neural network (RNN):** a small network (a GRU) that reads the last week of
  demand and weather hour by hour and outputs the forecast and its range for every hour of the
  next day.
- **Baselines:** the typical profile for the hour, weekday and time of year, and the same hour
  one week earlier. They set the bar that a model has to clear.
- **Perfect weather variants:** the same models given the weather that was later observed.
- **Whole-day scenarios:** 24-hour paths built on the RNN's forecasts, so that totals over
  the day (energy, peak) get honest ranges too. Independent hours, a Gaussian copula and a
  normalizing flow differ only in how they tie the hours together.

**Data.** Demand and REE's forecast from REE's ESIOS service; weather and archived weather
forecasts from Open-Meteo; Spanish public holidays. Demand is shown with the definition REE
used before December 2025 (without estimated self-consumption) throughout.

**Excluded days.** 28 and 29 April 2025 (the Iberian blackout) and 11 June 2026 (demand about
7 GW below REE's forecast in the evening, cause not investigated) are left out of all scores.

Dataset version `{meta["dataset_hash"]}`.
"""
        )


main()
