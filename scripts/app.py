import streamlit as st
import pandas as pd
from pandas import DateOffset
import plotly.express as px
import requests
import os
import json

TITLE    = "California Food Assistance Dashboard"
SUBTITLE = "CalFresh (SNAP) application rates by county, predicted using Google Trends and BLS unemployment data."
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

ALERT_LABELS_CSV    = os.path.join("outputs", "dashboard_v3", "historical_labels_v3.csv")
FORECAST_LABELS_CSV = os.path.join("outputs", "dashboard_v3", "forecast_labels_v3.csv")
PREDICTIONS_CSV     = os.path.join("outputs", "dashboard_v2", "predictions_v2.csv")

GENERAL_CAPTION = (
    "Colors compare each county's CalFresh application rate with the average of its two or three most "
    "recent available observations for the same calendar month (at least two are required). Red means the "
    "rate is unusually high for that time of year compared with the county's own history, Yellow means "
    "elevated, and Green means not flagged: below the county's Yellow cutoff for that month. Gray means a color "
    "could not be assigned because data or history are insufficient. Colors describe CalFresh applications, "
    "not food insecurity or food bank use directly."
)
FORECAST_DEMO_CAPTION = (
    "Retrospective research demonstration: This map presents a simulated forecast for January 2026 using "
    "the study's historical information cutoff of December 2025. It is not a current forecast or a validated "
    "operational warning system."
)
FORECAST_JANUARY_CAPTION = (
    "The model has tended to underestimate January demand in retrospective evaluations. Green does not "
    "guarantee that demand will remain within normal levels."
)
HISTORICAL_CAPTION = (
    "Historical colors show observed CalFresh application demand relative to each county's seasonal "
    "history, using alert thresholds calculated from data through December 2025. Months before March 2019 "
    "are Gray because at least two earlier observations for the same calendar month are required. Later "
    "months may also be Gray when data or sufficient historical deviations are unavailable. These are "
    "retrospective classifications, not forecasts made at the time."
)

FLAG_COLOR_MAP = {
    "Unusually high":  "#e74c3c",
    "Elevated":        "#f7ca18",
    "Not flagged":     "#27ae60",
    "Not enough data": "#888888",
}

DISPLAY_LABEL = {
    "Red":    "Unusually high",
    "Yellow": "Elevated",
    "Green":  "Not flagged",
    "Gray":   "Not enough data",
    None:     "Not enough data",
}

ALL_FLAGS = ["Unusually high", "Elevated", "Not flagged", "Not enough data"]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_snap_data():
    df = pd.read_csv(
        "src/data/SNAPApps/SNAPData.csv",
        header=None,
        names=["county", "date_str", "SNAP_Applications"],
        thousands=","
    )
    df["date"] = pd.to_datetime(df["date_str"].str.strip(), format="%b %Y", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
    )
    df["SNAP_Applications"] = pd.to_numeric(
        df["SNAP_Applications"].replace("*", pd.NA), errors="coerce"
    )
    return df


@st.cache_data
def load_pop_data():
    try:
        pop_df = pd.read_csv("src/data/popData.csv")
        pop_df.columns = pop_df.columns.str.strip()
        pop_df["county_clean"] = pop_df["County"].str.replace(" County", "", regex=False)
        return pop_df
    except Exception as e:
        st.warning(f"Could not load population data: {e}")
        return None


@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson["features"]:
        name = feature["properties"].get("name", "")
        feature["properties"]["name"] = name.replace(" County", "").strip()
    return geojson


@st.cache_data
def load_alert_labels():
    """Returns two dicts keyed by (county, 'YYYY-MM-DD'): label (Green/Yellow/Red/Gray) and
    seasonal hover line, from outputs/dashboard_v3/historical_labels_v3.csv."""
    if not os.path.exists(ALERT_LABELS_CSV):
        return {}, {}
    df = pd.read_csv(ALERT_LABELS_CSV, parse_dates=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    keys = list(zip(df["county"], df["date_str"]))
    return dict(zip(keys, df["label"])), dict(zip(keys, df["hover_text"]))


@st.cache_data
def load_predictions():
    """Returns the v2 forecast (one row per county), or None if the file is missing."""
    if not os.path.exists(PREDICTIONS_CSV):
        return None
    df = pd.read_csv(PREDICTIONS_CSV)
    df["county"] = df["county"].astype(str)
    df["county_clean"] = df["county"].str.replace(" County", "", regex=False)
    return df


@st.cache_data
def load_forecast_labels():
    """Returns dicts county -> label and county -> seasonal hover line for the forecast month."""
    if not os.path.exists(FORECAST_LABELS_CSV):
        return {}, {}
    df = pd.read_csv(FORECAST_LABELS_CSV)
    counties = df["county"].astype(str).str.replace(" County", "", regex=False)
    return dict(zip(counties, df["label"])), dict(zip(counties, df["hover_text"]))


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_snap(val):
    try:
        if pd.isnull(val):
            return "No data"
        v = float(val)
        return f"{int(v):,}"
    except Exception:
        return str(val)


def get_historical_label(county, date, alert_labels):
    """Look up label for a historical county-month. Gray if not in the label file."""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    return alert_labels.get((county, date_str), "Gray")


def get_historical_hover(county, date, alert_hover):
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    return alert_hover.get((county, date_str))


def forecast_flags(pred_df, forecast_labels):
    """Forecast-month flag per prediction row, read from forecast_labels_v3.csv (Gray if absent)."""
    return pred_df["county_clean"].map(lambda c: forecast_labels.get(c, "Gray"))


def add_pop_columns(df, pop_df):
    """Merge population info into df on county_clean."""
    if pop_df is None:
        return df
    df["county_clean"] = df["county"].str.replace(" County", "", regex=False)
    return df.merge(
        pop_df[["county_clean", "metro_area", "Population", "Population Density", "fips"]],
        on="county_clean", how="left"
    )


def build_hover(row, snap_label, snap_val, pred_val=None, pred_month=None, seasonal_line=None):
    pop   = f"{int(row['Population']):,}"   if "Population" in row and pd.notna(row.get("Population")) else "N/A"
    dens  = f"{int(row['Population Density']):,}" if "Population Density" in row and pd.notna(row.get("Population Density")) else "N/A"
    metro = row.get("metro_area", "N/A")
    fips  = row.get("fips", "N/A")
    text  = (f"<b>{row['county']}</b><br>"
             f"Metro: {metro}<br>"
             f"Population: {pop}<br>"
             f"Population Density: {dens}<br>"
             f"FIPS: {fips}<br>"
             f"{snap_label}: {snap_val}")
    if pred_val is not None and pred_month is not None:
        text += f"<br>Predicted for {pred_month}: {pred_val}"
    if isinstance(seasonal_line, str) and seasonal_line:
        text += f"<br>{seasonal_line}"
    return text


def draw_map(filtered_df, title_note=""):
    counties_geojson = load_geojson()
    plot_df = filtered_df.copy()

    # Remap internal flag names to display labels
    plot_df["Flag"] = plot_df["Flag"].map(lambda f: DISPLAY_LABEL.get(f, "Not enough data"))

    # Add one dummy row per missing category so all legend entries always appear
    present = set(plot_df["Flag"])
    dummy_rows = []
    for flag in ALL_FLAGS:
        if flag not in present:
            dummy_rows.append({"county": "", "Flag": flag, "hover_text": ""})
    if dummy_rows:
        plot_df = pd.concat([plot_df, pd.DataFrame(dummy_rows)], ignore_index=True)

    fig = px.choropleth(
        plot_df,
        geojson=counties_geojson,
        locations="county",
        color="Flag",
        color_discrete_map=FLAG_COLOR_MAP,
        category_orders={"Flag": ALL_FLAGS},
        custom_data=["hover_text"],
        featureidkey="properties.name",
        scope="usa",
        height=680,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        showlegend=True,
        legend=dict(
            title=None,
            orientation="h",
            x=0.0, y=1.08,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        ),
        title=dict(text=title_note, x=0.5, y=1.0, yanchor="top", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title=TITLE, layout="wide")

# Load shared resources once
snap_df        = load_snap_data()
pop_df         = load_pop_data()
alert_labels, alert_hover = load_alert_labels()
forecast_labels, forecast_hover = load_forecast_labels()

tabs = st.tabs(["Historical Map", "Predictions Map"])


# ── Tab 1: Historical Map ─────────────────────────────────────────────────────

with tabs[0]:
    st.title(TITLE)
    st.markdown(f"<h4 style='margin-top:-12px;color:gray'>{SUBTITLE}</h4>",
                unsafe_allow_html=True)

    unique_dates  = snap_df["date"].sort_values().unique()
    date_options  = [pd.Timestamp(d).strftime("%b %Y") for d in unique_dates]
    selected_date = st.selectbox("Select Month", options=date_options,
                                 index=len(date_options) - 1)

    sel_dt       = pd.to_datetime(selected_date, format="%b %Y")
    filtered_df  = snap_df[snap_df["date"].dt.strftime("%b %Y") == selected_date].copy()
    filtered_df  = add_pop_columns(filtered_df, pop_df)

    # Flag from alert labels (Gray if pre-model or missing)
    filtered_df["Flag"] = filtered_df.apply(
        lambda row: get_historical_label(row["county"], row["date"], alert_labels), axis=1
    )

    filtered_df["hover_text"] = filtered_df.apply(
        lambda row: build_hover(row,
                                snap_label="SNAP Applications",
                                snap_val=format_snap(row["SNAP_Applications"]),
                                seasonal_line=get_historical_hover(row["county"], row["date"],
                                                                   alert_hover)),
        axis=1,
    )

    draw_map(filtered_df, title_note=selected_date)
    st.caption(GENERAL_CAPTION)
    st.caption(HISTORICAL_CAPTION)


# ── Tab 2: Predictions Map ────────────────────────────────────────────────────

with tabs[1]:
    st.title("Predictions")

    pred_df   = load_predictions()
    pred_month_english = None

    if pred_df is not None:
        if not pred_df.empty:
            # target_month in the CSV is the target SNAP month.
            pred_date          = pred_df["target_month"].iloc[0]
            pred_dt            = pd.to_datetime(pred_date)
            target_month       = pred_dt
            pred_month_english = target_month.strftime("%b %Y")

    # ── Prominent prediction target banner ────────────────────────────────────
    if pred_month_english:
        st.info(
            f"**Simulated forecast for {pred_month_english}** (retrospective demonstration using data through Dec 2025). Select \"Predicted — {pred_month_english}\" at the bottom of the dropdown to view it."
        )
    else:
        st.warning("No prediction file found. Run scripts/dashboard_v2/part_b_build_dashboard.py "
                   "to generate the v2 forecast.")

    unique_dates = snap_df["date"].sort_values().unique()
    date_options = [pd.Timestamp(d).strftime("%b %Y") for d in unique_dates]

    if pred_month_english:
        pred_option           = f"Predicted — {pred_month_english}"
        date_options_extended = date_options + [pred_option]
    else:
        date_options_extended = date_options

    selected_date = st.selectbox("Select Month", options=date_options_extended,
                                 index=len(date_options_extended) - 1, key="pred_month")

    use_predicted = (pred_df is not None and pred_month_english is not None
                     and selected_date == f"Predicted — {pred_month_english}")

    if use_predicted:
        # ── Predicted month: labels come from forecast_labels_v3.csv ─────────
        base_df = pred_df.copy()
        base_df = base_df.drop(columns=[c for c in ["metro_area", "Population"] if c in base_df.columns])
        base_df = add_pop_columns(base_df, pop_df)

        base_df["Flag"] = forecast_flags(base_df, forecast_labels)

        base_df["hover_text"] = base_df.apply(
            lambda row: build_hover(
                row,
                snap_label=f"Predicted applications ({pred_month_english})",
                snap_val=format_snap(row.get("predicted_applications")),
                seasonal_line=forecast_hover.get(row["county_clean"]),
            ),
            axis=1,
        )
        draw_map(base_df, title_note=f"Predicted — {pred_month_english}")
        st.caption(GENERAL_CAPTION)
        st.caption(FORECAST_DEMO_CAPTION)
        st.caption(FORECAST_JANUARY_CAPTION)

    else:
        # ── Historical month: same alert labels as Tab 1, plus prediction in hover ──
        filtered_df = snap_df[snap_df["date"].dt.strftime("%b %Y") == selected_date].copy()
        filtered_df = add_pop_columns(filtered_df, pop_df)

        filtered_df["Flag"] = filtered_df.apply(
            lambda row: get_historical_label(row["county"], row["date"], alert_labels), axis=1
        )

        # Add predicted applications in hover if prediction file exists
        if pred_df is not None and pred_month_english:
            filtered_df["county_clean"] = filtered_df["county"].str.replace(" County", "", regex=False)
            pred_map = pred_df.set_index("county_clean")["predicted_applications"].to_dict()
            filtered_df["hover_text"] = filtered_df.apply(
                lambda row: build_hover(
                    row,
                    snap_label="SNAP Applications",
                    snap_val=format_snap(row["SNAP_Applications"]),
                    pred_val=format_snap(pred_map.get(
                        row["county"].replace(" County", "").strip()
                    )),
                    pred_month=pred_month_english,
                    seasonal_line=get_historical_hover(row["county"], row["date"], alert_hover),
                ),
                axis=1,
            )
        else:
            filtered_df["hover_text"] = filtered_df.apply(
                lambda row: build_hover(row,
                                        snap_label="SNAP Applications",
                                        snap_val=format_snap(row["SNAP_Applications"]),
                                        seasonal_line=get_historical_hover(row["county"], row["date"],
                                                                           alert_hover)),
                axis=1,
            )

        draw_map(filtered_df, title_note=selected_date)
        st.caption(GENERAL_CAPTION)
        st.caption(HISTORICAL_CAPTION)
