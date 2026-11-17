"""Dashboard: day-ahead forecasts of Spanish electricity demand against what happened.

Run locally with ``make app`` (or ``streamlit run app/streamlit_app.py``). The app reads only
the small bundle in ``app/data/`` (built by ``python -m loadforecast.viz.app_data``), so it
does not need the project package or the raw data.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    "rnn_forecast": ("Neural network (GRU)", "#c8303c"),
    "climatology": ("Baseline: typical profile", "#eda100"),
    "naive_weekly": ("Baseline: same hour last week", "#e87ba4"),
    "lgbm_oracle": ("LightGBM, perfect weather", "#008300"),
    "bayes_nuts_oracle": ("Bayesian model, perfect weather", "#6250d6"),
    "rnn_oracle": ("Neural network, perfect weather", "#8a6a1f"),
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
    day, accuracy, calibration, about = st.tabs(
        ["Explore a day", "Accuracy over the year", "Are the ranges honest?", "About"]
    )
    with day:
        day_tab(actuals, predictions, meta)
    with accuracy:
        accuracy_tab(scores)
    with calibration:
        calibration_tab(meta)
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
- **Neural network (GRU):** a recurrent network that reads the last week of demand and weather
  hour by hour and outputs the forecast and its range for every hour of the next day.
- **Baselines:** the typical profile for the hour, weekday and time of year, and the same hour
  one week earlier. They set the bar that a model has to clear.
- **Perfect weather variants:** the same models given the weather that was later observed.

**Data.** Demand and REE's forecast from REE's ESIOS service; weather and archived weather
forecasts from Open-Meteo; Spanish public holidays. Demand is shown with the definition REE
used before December 2025 (without estimated self-consumption) throughout.

**Excluded days.** 28 and 29 April 2025 (the Iberian blackout) and 11 June 2026 (demand about
7 GW below REE's forecast in the evening, cause not investigated) are left out of all scores.

Dataset version `{meta["dataset_hash"]}`.
"""
        )


main()
