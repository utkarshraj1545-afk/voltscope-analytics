import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from collections import Counter

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="VoltScope | EV Operations Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}
.stApp {
    background: #0a1018;
    color: #e8eef5;
}
[data-testid="stSidebar"] {
    background: #101a26;
    border-right: 1px solid #26384a;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#142333,#101a26);
    border: 1px solid #26394c;
    padding: 16px 18px;
    border-radius: 13px;
}
[data-testid="stMetricLabel"] {
    color: #9db0c2 !important;
    font-size: 0.82rem;
}
[data-testid="stMetricValue"] {
    color: #f2f7fb !important;
    font-weight: 800;
}
h1, h2, h3 {
    letter-spacing: -0.035em;
}
h1 {
    color: #f3f7fb;
    font-weight: 800;
}
h2 {
    color: #e8f0f7;
    font-weight: 750;
}
h3 {
    color: #d9e5ef;
    font-weight: 700;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.kicker {
    color: #42d6b5;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: .72rem;
    font-weight: 800;
}
.hero {
    padding: 22px 26px;
    border: 1px solid #26394c;
    border-radius: 18px;
    background: radial-gradient(
        circle at 85% 15%,
        #163d48 0%,
        #142333 34%,
        #101a26 78%
    );
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f4f8fb;
    margin: 4px 0;
}
.hero-sub {
    color: #a9bdce;
    font-size: .94rem;
}
.small-muted {
    color: #8fa4b8;
    font-size: .82rem;
}
hr {
    border-color: #26394c;
}
</style>
""", unsafe_allow_html=True)

COLORS = [
    "#35d0ba", "#61a8ff", "#f5b84b",
    "#ed7185", "#a78bfa", "#8bd17c", "#f08f54"
]

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dce7f0", family="Manrope"),
    margin=dict(l=12, r=12, t=42, b=12),
    xaxis=dict(gridcolor="#243444", zerolinecolor="#243444"),
    yaxis=dict(gridcolor="#243444", zerolinecolor="#243444"),
    legend=dict(orientation="h", y=-0.22)
)


def style_fig(fig, height=340):
    fig.update_layout(**PLOT, height=height)
    return fig


def money_cr(value):
    return f"₹{value / 1e7:,.2f} Cr"


def canonical_city(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    mapping = {
        "blr": "Bengaluru",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "mum": "Mumbai",
        "bombay": "Mumbai",
        "mumbai": "Mumbai",
        "hyd": "Hyderabad",
        "hyd.": "Hyderabad",
        "hyderabad": "Hyderabad",
        "jai": "Jaipur",
        "jaipur": "Jaipur",
        "pun": "Pune",
        "pune": "Pune",
        "delhi": "Delhi NCR",
        "new delhi": "Delhi NCR",
        "delhi ncr": "Delhi NCR",
        "ncr": "Delhi NCR"
    }

    return mapping.get(value, str(value).title())


def style_section(kicker, title, subtitle=None):
    st.markdown(
        f'<div class="kicker">{kicker}</div><h2>{title}</h2>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.caption(subtitle)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# ============================================================
# DATA FILE DISCOVERY
# ============================================================
def locate_files(root):
    filenames = [
        "swap_events.csv",
        "stations.csv",
        "riders.csv",
        "batteries.csv",
        "support_tickets.csv",
        "station_hourly_status.csv",
        "city_daily_context.csv"
    ]

    found = {}

    for filename in filenames:
        matches = list(root.rglob(filename))
        if matches:
            found[filename] = matches[0]

    return found


def find_data_root():
    required = {
        "swap_events.csv",
        "stations.csv",
        "riders.csv",
        "batteries.csv",
        "support_tickets.csv"
    }

    candidates = [
        Path("data"),
        Path("dataset"),
        Path("."),
        Path("Hackathon Data Set")
    ]

    for folder in candidates:
        if folder.exists():
            found = {
                p.name for p in folder.rglob("*.csv")
            }
            if required.issubset(found):
                return folder

    for folder in Path(".").iterdir():
        if folder.is_dir():
            found = {
                p.name for p in folder.rglob("*.csv")
            }
            if required.issubset(found):
                return folder

    return None


# ============================================================
# LOAD REFERENCE DATA
# ============================================================
@st.cache_data(show_spinner="Reading reference datasets...")
def load_reference_data(root_str):
    root = Path(root_str)
    files = locate_files(root)

    required = [
        "stations.csv",
        "riders.csv",
        "batteries.csv",
        "support_tickets.csv"
    ]

    missing = [
        name for name in required
        if name not in files
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing)
        )

    stations = pd.read_csv(files["stations.csv"])
    riders = pd.read_csv(files["riders.csv"])
    batteries = pd.read_csv(files["batteries.csv"])
    tickets = pd.read_csv(files["support_tickets.csv"])

    stations["city_clean"] = (
        stations["city"].map(canonical_city)
    )

    if "created_ts" in tickets.columns:
        tickets["created_ts"] = pd.to_datetime(
            tickets["created_ts"],
            errors="coerce"
        )

    if "retired_date" in batteries.columns:
        batteries["is_retired"] = (
            batteries["retired_date"].notna()
        )
    else:
        batteries["is_retired"] = False

    return files, stations, riders, batteries, tickets


# ============================================================
# PROCESS SWAP EVENTS IN CHUNKS
# ============================================================
@st.cache_data(show_spinner="Processing swap events in chunks...")
def process_swaps(path_str, station_lookup):
    chunksize = 250_000

    overall = Counter()

    station_parts = []
    hour_parts = []
    city_parts = []
    city_hour_parts = []
    rider_parts = []
    event_parts = []

    usecols = [
        "rider_id",
        "station_id",
        "event_ts",
        "event_type",
        "queue_wait_sec",
        "amount_charged_inr"
    ]

    for chunk in pd.read_csv(
        path_str,
        usecols=usecols,
        chunksize=chunksize
    ):
        chunk["event_ts"] = pd.to_datetime(
            chunk["event_ts"],
            errors="coerce"
        )

        chunk["hour"] = chunk["event_ts"].dt.hour

        chunk["completed"] = (
            chunk["event_type"] == "swap_completed"
        )

        chunk["battery_shortage"] = (
            chunk["event_type"] == "failed_no_charged_battery"
        )

        chunk["revenue"] = np.where(
            chunk["completed"],
            chunk["amount_charged_inr"].fillna(0),
            0
        )

        overall["attempts"] += len(chunk)
        overall["completed"] += int(chunk["completed"].sum())
        overall["revenue"] += float(chunk["revenue"].sum())
        overall["shortages"] += int(
            chunk["battery_shortage"].sum()
        )

        event_parts.append(
            chunk["event_type"].value_counts()
        )

        station_parts.append(
            chunk.groupby("station_id").agg(
                attempts=("event_type", "size"),
                completed=("completed", "sum"),
                shortages=("battery_shortage", "sum"),
                queue_sum=("queue_wait_sec", "sum"),
                queue_n=("queue_wait_sec", "count"),
                revenue=("revenue", "sum")
            ).reset_index()
        )

        hour_parts.append(
            chunk.groupby("hour").agg(
                attempts=("event_type", "size"),
                completed=("completed", "sum"),
                shortages=("battery_shortage", "sum")
            ).reset_index()
        )

        city_chunk = chunk.merge(
            station_lookup,
            on="station_id",
            how="left"
        )

        city_chunk["city_clean"] = (
            city_chunk["city_clean"].fillna("Unknown")
        )

        city_parts.append(
            city_chunk.groupby("city_clean").agg(
                attempts=("event_type", "size"),
                completed=("completed", "sum"),
                shortages=("battery_shortage", "sum"),
                revenue=("revenue", "sum")
            ).reset_index()
        )

        city_hour_parts.append(
            city_chunk.groupby(
                ["city_clean", "hour"]
            ).agg(
                attempts=("event_type", "size"),
                completed=("completed", "sum")
            ).reset_index()
        )

        rider_parts.append(
            chunk.groupby("rider_id").agg(
                rider_attempts=("event_type", "size"),
                rider_completed=("completed", "sum"),
                rider_revenue=("revenue", "sum")
            ).reset_index()
        )

    def combine(parts, key):
        data = pd.concat(parts, ignore_index=True)
        numeric = [
            col for col in data.columns
            if col != key
        ]
        return data.groupby(
            key, as_index=False
        )[numeric].sum()

    station = combine(station_parts, "station_id")

    station["failure_rate_pct"] = (
        1 - station["completed"] / station["attempts"]
    ) * 100

    station["shortage_rate_pct"] = (
        station["shortages"] / station["attempts"]
    ) * 100

    station["avg_queue_wait_sec"] = (
        station["queue_sum"] /
        station["queue_n"].replace(0, np.nan)
    )

    station["city_clean"] = station["station_id"].map(
        station_lookup.set_index("station_id")["city_clean"]
    )

    hourly = combine(hour_parts, "hour").sort_values("hour")

    hourly["failure_rate_pct"] = (
        1 - hourly["completed"] / hourly["attempts"]
    ) * 100

    city = combine(city_parts, "city_clean")

    city["failure_rate_pct"] = (
        1 - city["completed"] / city["attempts"]
    ) * 100

    city["shortage_rate_pct"] = (
        city["shortages"] / city["attempts"]
    ) * 100

    city_hourly = (
        pd.concat(city_hour_parts, ignore_index=True)
        .groupby(
            ["city_clean", "hour"],
            as_index=False
        )[["attempts", "completed"]]
        .sum()
    )

    rider_activity = combine(rider_parts, "rider_id")

    event_counts = (
        pd.concat(event_parts)
        .groupby(level=0)
        .sum()
        .rename_axis("event_type")
        .reset_index(name="count")
    )

    return {
        "kpis": dict(overall),
        "station": station,
        "hourly": hourly,
        "city": city,
        "city_hourly": city_hourly,
        "rider_activity": rider_activity,
        "event_counts": event_counts
    }


# ============================================================
# PROCESS TELEMETRY
# ============================================================
@st.cache_data(show_spinner="Summarizing station telemetry...")
def process_telemetry(path_str):
    parts = []

    usecols = [
        "station_id",
        "charged_2w_min",
        "packs_charging",
        "packs_quarantined",
        "outage_minutes",
        "chargers_online"
    ]

    for chunk in pd.read_csv(
        path_str,
        usecols=usecols,
        chunksize=250_000
    ):
        parts.append(
            chunk.groupby("station_id").agg(
                avg_charged_2w=("charged_2w_min", "mean"),
                avg_packs_charging=("packs_charging", "mean"),
                avg_packs_quarantined=("packs_quarantined", "mean"),
                outage_minutes=("outage_minutes", "sum"),
                avg_chargers_online=("chargers_online", "mean")
            ).reset_index()
        )

    data = pd.concat(parts, ignore_index=True)

    return data.groupby(
        "station_id", as_index=False
    ).mean(numeric_only=True)


# ============================================================
# SIDEBAR AND DATA LOADING
# ============================================================
st.sidebar.markdown("## ⚡ VOLTSCOPE")
st.sidebar.caption("EV Operations Intelligence")
st.sidebar.markdown("---")

with st.sidebar.expander("Data location", expanded=False):
    st.write(
        "Place the extracted CSV folder beside app.py, "
        "or enter its path below."
    )
    data_path = st.text_input(
        "Dataset folder",
        value="."
    )


# ---------------- DEPLOYMENT DATA LOADING ----------------
DATA_DIR = Path(__file__).resolve().parent

try:
    stations = pd.read_csv(DATA_DIR / "stations.csv")
    riders = pd.read_csv(DATA_DIR / "riders.csv")
    batteries = pd.read_csv(DATA_DIR / "batteries.csv")
    tickets = pd.read_csv(DATA_DIR / "tickets.csv")

    swaps = {
        "station": pd.read_csv(DATA_DIR / "station.csv"),
        "hourly": pd.read_csv(DATA_DIR / "hourly.csv"),
        "city": pd.read_csv(DATA_DIR / "city.csv"),
        "city_hourly": pd.read_csv(DATA_DIR / "city_hourly.csv"),
        "rider_activity": pd.read_csv(DATA_DIR / "rider_activity.csv"),
        "event_counts": pd.read_csv(DATA_DIR / "event_counts.csv")
    }

    city_totals = swaps["city"]

    swaps["kpis"] = {
        "attempts": int(city_totals["attempts"].sum()),
        "completed": int(city_totals["completed"].sum()),
        "revenue": float(city_totals["revenue"].sum()),
        "shortages": int(city_totals["shortages"].sum())
    }

    telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")

    station = swaps["station"].merge(
        stations[
            [
                "station_id",
                "city_clean",
                "slots_2w",
                "slots_3w",
                "location_type",
                "host_type"
            ]
        ],
        on="station_id",
        how="left",
        suffixes=("", "_master")
    )

    station["city_clean"] = station["city_clean"].fillna(
        station["city_clean_master"]
    )

    station = station.merge(
        telemetry,
        on="station_id",
        how="left"
    )

    station_lookup = stations[
        ["station_id", "city_clean"]
    ].drop_duplicates("station_id")

except Exception as error:
    st.error(f"Could not load deployment data: {error}")
    st.stop()


# NAVIGATION AND CITY FILTER
# ============================================================
pages = [
    "Executive Overview",
    "City & Station Performance",
    "Battery Availability",
    "Battery Health",
    "Rider & Revenue",
    "Customer Support"
]

page = st.sidebar.radio(
    "NAVIGATION",
    pages,
    index=0
)

cities = sorted(
    station["city_clean"].dropna().unique()
)

selected_cities = st.sidebar.multiselect(
    "Filter cities",
    cities,
    default=cities
)

st.sidebar.markdown("---")
st.sidebar.caption("DATA WINDOW · Jan 2024 – Jun 2025")
st.sidebar.caption("Analysis mode · Descriptive analytics")

station_f = station[
    station["city_clean"].isin(selected_cities)
].copy()

city_f = swaps["city"][
    swaps["city"]["city_clean"].isin(selected_cities)
].copy()

# Station-level swap records are the source for operating-city filtering.
filtered_attempts = int(city_f["attempts"].sum())
filtered_completed = int(city_f["completed"].sum())
filtered_shortages = int(city_f["shortages"].sum())
filtered_revenue = float(city_f["revenue"].sum())

filtered_failure_rate = (
    (filtered_attempts - filtered_completed) /
    filtered_attempts * 100
    if filtered_attempts else 0
)

filtered_shortage_rate = (
    filtered_shortages / filtered_attempts * 100
    if filtered_attempts else 0
)

# Main dashboard banner
st.markdown(
    '<div class="hero">'
    '<div class="kicker">FLEET PERFORMANCE / LIVE ANALYTICS WORKSPACE</div>'
    '<div class="hero-title">VoltScope Analytics '
    '<span style="color:#35d0ba">↗</span></div>'
    '<div class="hero-sub">A unified view of swap reliability, '
    'battery readiness, customer experience and fleet economics.</div>'
    '</div>',
    unsafe_allow_html=True
)
# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":

    style_section(
        "01 / COMMAND CENTER",
        "Executive Overview",
        "A concise operating picture across the selected cities."
    )

    metric_row([
        ("SWAP ATTEMPTS", f"{filtered_attempts:,}", None),
        (
            "COMPLETED SWAPS",
            f"{filtered_completed:,}",
            f"{(filtered_completed / filtered_attempts * 100) if filtered_attempts else 0:.1f}% completion"
        ),
        ("FAILURE RATE", f"{filtered_failure_rate:.2f}%", None),
        ("RECORDED REVENUE", money_cr(filtered_revenue), None)
    ])

    st.markdown("")

    metric_row([
        (
            "BATTERY SHORTAGES",
            f"{filtered_shortages:,}",
            f"{filtered_shortage_rate:.2f}% of attempts"
        ),
        ("STATIONS", f"{len(station_f):,}", None),
        ("BATTERIES IN FLEET", f"{len(batteries):,}", "Network-wide"),
        ("SUPPORT TICKETS", f"{len(tickets):,}", "Network-wide")
    ])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### City reliability")

        fig = px.bar(
            city_f.sort_values("attempts", ascending=False),
            x="city_clean",
            y="failure_rate_pct",
            color="failure_rate_pct",
            color_continuous_scale=[
                "#35d0ba", "#f5b84b", "#ed7185"
            ],
            hover_data=["attempts", "shortages"],
            labels={
                "city_clean": "City",
                "failure_rate_pct": "Failure rate (%)"
            }
        )

        st.plotly_chart(
            style_fig(fig, 360),
            use_container_width=True
        )

    with c2:
        st.markdown("### Swap outcome mix — network-wide")

        fig = px.pie(
            swaps["event_counts"],
            names="event_type",
            values="count",
            hole=0.68,
            color_discrete_sequence=COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        st.plotly_chart(
            style_fig(fig, 360),
            use_container_width=True
        )

    st.markdown("### Operating priorities")

    top = station_f.sort_values(
        ["failure_rate_pct", "attempts"],
        ascending=False
    ).head(10)

    st.dataframe(
        top[
            [
                "station_id",
                "city_clean",
                "attempts",
                "completed",
                "shortages",
                "failure_rate_pct",
                "shortage_rate_pct",
                "avg_queue_wait_sec"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Swap, shortage and revenue KPIs reflect selected station cities. "
        "Battery fleet and support-ticket totals are network-wide. "
        "Revenue is recorded revenue, not profit."
    )


# ============================================================
# PAGE 2: CITY & STATION PERFORMANCE
# ============================================================
elif page == "City & Station Performance":

    style_section(
        "02 / NETWORK RELIABILITY",
        "City & Station Performance",
        "Compare service reliability, demand and station-level pressure."
    )

    d = city_f.sort_values(
        "attempts",
        ascending=False
    ).copy()

    d["failed"] = d["attempts"] - d["completed"]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Completed vs failed swaps")

        outcome = d.melt(
            id_vars=["city_clean"],
            value_vars=["completed", "failed"],
            var_name="outcome",
            value_name="swaps"
        )

        fig = px.bar(
            outcome,
            x="city_clean",
            y="swaps",
            color="outcome",
            barmode="stack",
            color_discrete_map={
                "completed": "#35d0ba",
                "failed": "#ed7185"
            },
            labels={
                "city_clean": "City",
                "swaps": "Swap attempts",
                "outcome": "Outcome"
            }
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    with c2:
        st.markdown("### City failure rate")

        fig = px.scatter(
            d,
            x="attempts",
            y="failure_rate_pct",
            size="shortages",
            color="city_clean",
            hover_name="city_clean",
            labels={
                "attempts": "Swap attempts",
                "failure_rate_pct": "Failure rate (%)"
            }
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    st.markdown("### Station leaderboard")

    sort_by = st.selectbox(
        "Rank stations by",
        [
            "Failure rate",
            "Battery shortages",
            "Swap attempts",
            "Queue wait"
        ]
    )

    colmap = {
        "Failure rate": "failure_rate_pct",
        "Battery shortages": "shortages",
        "Swap attempts": "attempts",
        "Queue wait": "avg_queue_wait_sec"
    }

    eligible = station_f[
        station_f["attempts"] >= 5000
    ].copy()

    top = eligible.sort_values(
        colmap[sort_by],
        ascending=False
    ).head(25)

    st.caption(
        f"Showing stations with at least 5,000 swap attempts "
        f"({len(eligible)} stations qualify)."
    )

    st.dataframe(
        top[
            [
                "station_id",
                "city_clean",
                "attempts",
                "completed",
                "shortages",
                "failure_rate_pct",
                "shortage_rate_pct",
                "avg_queue_wait_sec"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Demand by recorded event hour")

    hourly = swaps["city_hourly"]

    hourly = hourly[
        hourly["city_clean"].isin(selected_cities)
    ].groupby(
        "hour",
        as_index=False
    )["attempts"].sum().sort_values("hour")

    fig = px.line(
        hourly,
        x="hour",
        y="attempts",
        markers=True,
        labels={
            "hour": "Hour in event timestamp",
            "attempts": "Swap attempts"
        }
    )

    fig.update_traces(
        line=dict(color="#35d0ba", width=3)
    )

    st.plotly_chart(
        style_fig(fig, 300),
        use_container_width=True
    )

    st.caption(
        "Hour-of-day uses event_ts as stored. Validate the documented "
        "firmware timezone correction before interpreting these as local-time peaks."
    )


# ============================================================
# PAGE 3: BATTERY AVAILABILITY
# ============================================================
elif page == "Battery Availability":

    style_section(
        "03 / BATTERY READINESS",
        "Battery Availability",
        "Understand charged inventory, shortage pressure and replenishment needs."
    )

    if telemetry.empty:
        st.warning(
            "Station telemetry file not found or could not be processed. "
            "Showing swap-derived shortage metrics only."
        )

    metric_row([
        (
            "BATTERY SHORTAGES",
            f"{filtered_shortages:,}",
            "Selected cities"
        ),
        (
            "SHORTAGE RATE",
            f"{filtered_shortage_rate:.2f}%",
            "Selected cities"
        ),
        (
            "STATIONS IN VIEW",
            f"{len(station_f):,}",
            None
        )
    ])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Charged inventory vs shortage rate")

        if "avg_charged_2w" in station_f.columns:
            available = station_f.dropna(
                subset=[
                    "avg_charged_2w",
                    "shortage_rate_pct"
                ]
            )

            fig = px.scatter(
                available,
                x="avg_charged_2w",
                y="shortage_rate_pct",
                size="attempts",
                color="city_clean",
                hover_name="station_id",
                labels={
                    "avg_charged_2w": "Average charged 2W minimum",
                    "shortage_rate_pct": "Shortage rate (%)"
                }
            )

            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )
        else:
            st.info("Charged inventory telemetry is unavailable.")

    with c2:
        st.markdown("### Stations with most battery shortages")

        shortage_top = station_f.sort_values(
            "shortages",
            ascending=False
        ).head(15).sort_values("shortages")

        fig = px.bar(
            shortage_top,
            x="shortages",
            y="station_id",
            orientation="h",
            color="shortage_rate_pct",
            color_continuous_scale=[
                "#35d0ba", "#f5b84b", "#ed7185"
            ],
            labels={
                "station_id": "Station",
                "shortages": "Battery shortages",
                "shortage_rate_pct": "Shortage rate (%)"
            }
        )

        st.plotly_chart(
            style_fig(fig, 420),
            use_container_width=True
        )

    st.markdown("### Station battery readiness")

    columns = [
        "station_id",
        "city_clean",
        "attempts",
        "shortages",
        "shortage_rate_pct",
        "avg_charged_2w",
        "avg_packs_charging",
        "avg_packs_quarantined",
        "outage_minutes"
    ]

    columns = [
        col for col in columns
        if col in station_f.columns
    ]

    st.dataframe(
        station_f.sort_values(
            "shortage_rate_pct",
            ascending=False
        )[columns].head(30).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Charged inventory showed a weak negative linear correlation "
        "with shortage rate in the filtered station-hour sample (r = −0.068). "
        "Grouped shortage rates showed a clearer descriptive gradient. "
        "Charging-pack count alone was not a strong indicator."
    )


# ============================================================
# PAGE 4: BATTERY HEALTH
# ============================================================
elif page == "Battery Health":

    style_section(
        "04 / ASSET HEALTH",
        "Battery Health",
        "Track state of health, retirement concentration and supplier quality signals."
    )

    retired = int(batteries["is_retired"].sum())
    total_batteries = len(batteries)

    retirement_pct = (
        retired / total_batteries * 100
        if total_batteries else 0
    )

    metric_row([
        ("BATTERY FLEET", f"{total_batteries:,}", "Network-wide"),
        (
            "RECORDED RETIREMENTS",
            f"{retired:,}",
            f"{retirement_pct:.1f}% of fleet"
        ),
        (
            "AVERAGE CURRENT SOH",
            f"{batteries['current_soh_pct'].mean():.1f}%",
            None
        ),
        (
            "MEDIAN CURRENT SOH",
            f"{batteries['current_soh_pct'].median():.1f}%",
            None
        )
    ])

    supplier = batteries.groupby(
        "supplier",
        as_index=False
    ).agg(
        batteries=("battery_id", "count"),
        avg_soh=("current_soh_pct", "mean"),
        retired=("is_retired", "sum")
    )

    supplier["retirement_rate_pct"] = (
        supplier["retired"] / supplier["batteries"] * 100
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            supplier,
            x="supplier",
            y="avg_soh",
            color="avg_soh",
            color_continuous_scale=[
                "#ed7185", "#f5b84b", "#35d0ba"
            ],
            labels={
                "avg_soh": "Average current SOH (%)"
            }
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    with c2:
        fig = px.histogram(
            batteries,
            x="current_soh_pct",
            nbins=24,
            color_discrete_sequence=["#61a8ff"],
            labels={
                "current_soh_pct": "Current SOH (%)"
            }
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    st.markdown("### Supplier retirement summary")

    st.dataframe(
        supplier.sort_values(
            "avg_soh"
        )[
            [
                "supplier",
                "batteries",
                "avg_soh",
                "retired",
                "retirement_rate_pct"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Manufacturing lot risk")

    lot = batteries.groupby(
        "manufacturing_lot",
        as_index=False
    ).agg(
        batteries=("battery_id", "count"),
        avg_soh=("current_soh_pct", "mean"),
        retired=("is_retired", "sum")
    )

    # Show lots with the highest retirement counts first.
    lot = lot[
        lot["batteries"] >= 20
    ].sort_values(
        ["retired", "avg_soh"],
        ascending=[False, True]
    ).head(15)

    st.dataframe(
        lot.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "All 1,438 recorded retirements are associated with supplier "
        "Kyron in this dataset. This is a quality signal for investigation, "
        "not proof of supplier causation. Missing retirement records are "
        "not automatically classified as active."
    )


# ============================================================
# PAGE 5: RIDER & REVENUE
# ============================================================
elif page == "Rider & Revenue":

    style_section(
        "05 / CUSTOMER ECONOMICS",
        "Rider & Revenue",
        "Recorded completed-swap revenue and rider segment contribution."
    )

    # Rider profile analysis uses rider home city.
    # This is distinct from the station operating-city filter.
    rider_data = swaps["rider_activity"].merge(
        riders,
        on="rider_id",
        how="left"
    )

    rider_data["home_city_clean"] = (
        rider_data["home_city"].map(canonical_city)
    )

    rider_filtered = rider_data[
        rider_data["home_city_clean"].isin(selected_cities)
    ].copy()

    rider_revenue = rider_filtered["rider_revenue"].sum()
    rider_completed = rider_filtered["rider_completed"].sum()

    active_riders = rider_filtered.loc[
        rider_filtered["rider_completed"] > 0,
        "rider_id"
    ].nunique()

    avg_revenue_per_swap = (
        rider_revenue / rider_completed
        if rider_completed else 0
    )

    metric_row([
        (
            "RECORDED REVENUE",
            money_cr(rider_revenue),
            "Rider home-city selection"
        ),
        ("ACTIVE RIDERS", f"{active_riders:,}", None),
        ("COMPLETED SWAPS", f"{rider_completed:,}", None),
        (
            "REVENUE / COMPLETED SWAP",
            f"₹{avg_revenue_per_swap:.2f}",
            None
        )
    ])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Revenue by vehicle class")

        segment = rider_filtered.groupby(
            "vehicle_class",
            as_index=False
        ).agg(
            revenue=("rider_revenue", "sum"),
            swaps=("rider_completed", "sum")
        )

        fig = px.pie(
            segment,
            names="vehicle_class",
            values="revenue",
            hole=0.62,
            color_discrete_sequence=COLORS
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    with c2:
        st.markdown("### Revenue by plan")

        segment = rider_filtered.groupby(
            "plan_type",
            as_index=False
        ).agg(
            revenue=("rider_revenue", "sum"),
            swaps=("rider_completed", "sum")
        )

        fig = px.bar(
            segment,
            x="plan_type",
            y="revenue",
            color="plan_type",
            labels={
                "revenue": "Recorded revenue (₹)"
            }
        )

        st.plotly_chart(
            style_fig(fig),
            use_container_width=True
        )

    st.markdown("### Revenue by rider home city")

    city_revenue = rider_filtered.groupby(
        "home_city_clean",
        as_index=False
    ).agg(
        revenue=("rider_revenue", "sum"),
        riders=("rider_id", "nunique")
    )

    fig = px.bar(
        city_revenue.sort_values(
            "revenue",
            ascending=False
        ),
        x="home_city_clean",
        y="revenue",
        labels={
            "home_city_clean": "Rider home city",
            "revenue": "Recorded revenue (₹)"
        }
    )

    st.plotly_chart(
        style_fig(fig, 300),
        use_container_width=True
    )

    st.caption(
        "The city filter on this page uses rider home city, not the city "
        "where a swap took place. Revenue is recorded completed-swap "
        "revenue, not profit."
    )


# ============================================================
# PAGE 6: CUSTOMER SUPPORT
# ============================================================
elif page == "Customer Support":

    style_section(
        "06 / CUSTOMER EXPERIENCE",
        "Customer Support",
        "Understand complaint drivers, support workload and recorded satisfaction."
    )

    # Link each ticket to the city of its station.
    ticket_data = tickets.merge(
        station_lookup,
        on="station_id",
        how="left"
    )

    ticket_data["city_clean"] = (
        ticket_data["city_clean"].fillna("Unknown")
    )

    # Filter station-linked tickets by selected station cities.
    ticket_filtered = ticket_data[
        ticket_data["city_clean"].isin(selected_cities)
    ].copy()

    # Keep tickets with missing station ID visible as a separate
    # network-wide data-quality count, not as a selected city.
    unassigned_tickets = int(
        ticket_data["station_id"].isna().sum()
    )

    total_tickets = len(ticket_filtered)

    resolved = int(
        (ticket_filtered["resolution_status"] == "resolved").sum()
    )

    csat = ticket_filtered["csat_score"].dropna()

    avg_resolution = (
        ticket_filtered["resolution_hours"].mean()
        if total_tickets else np.nan
    )

    resolution_pct = (
        resolved / total_tickets * 100
        if total_tickets else 0
    )

    avg_csat = csat.mean() if len(csat) else np.nan

    metric_row([
        ("SUPPORT TICKETS", f"{total_tickets:,}", "Selected station cities"),
        (
            "RESOLVED",
            f"{resolved:,}",
            f"{resolution_pct:.1f}% of selected tickets"
        ),
        (
            "AVG RESOLUTION",
            f"{avg_resolution:.1f} h" if pd.notna(avg_resolution) else "N/A",
            None
        ),
        (
            "RECORDED CSAT",
            f"{avg_csat:.2f}/5" if pd.notna(avg_csat) else "N/A",
            f"{len(csat):,} responses"
        )
    ])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Complaint categories")

        category = ticket_filtered.groupby(
            "category",
            as_index=False
        ).agg(
            tickets=("ticket_id", "count"),
            avg_csat=("csat_score", "mean"),
            avg_resolution_hours=("resolution_hours", "mean")
        )

        fig = px.bar(
            category.sort_values("tickets"),
            x="tickets",
            y="category",
            orientation="h",
            color="tickets",
            color_continuous_scale=[
                "#61a8ff", "#35d0ba"
            ],
            labels={
                "tickets": "Tickets",
                "category": "Complaint category"
            }
        )

        st.plotly_chart(
            style_fig(fig, 360),
            use_container_width=True
        )

    with c2:
        st.markdown("### Support channels")

        channel = ticket_filtered.groupby(
            "channel",
            as_index=False
        ).agg(
            tickets=("ticket_id", "count"),
            avg_resolution=("resolution_hours", "mean"),
            avg_csat=("csat_score", "mean")
        )

        fig = px.bar(
            channel,
            x="channel",
            y="tickets",
            color="channel",
            labels={
                "tickets": "Tickets"
            }
        )

        st.plotly_chart(
            style_fig(fig, 360),
            use_container_width=True
        )

    st.markdown("### Complaint category detail")

    st.dataframe(
        category.sort_values(
            "tickets",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Stations with the most support tickets")

    top = ticket_filtered.dropna(
        subset=["station_id"]
    ).groupby(
        "station_id",
        as_index=False
    ).agg(
        tickets=("ticket_id", "count"),
        avg_csat=("csat_score", "mean"),
        avg_resolution_hours=("resolution_hours", "mean")
    )

    top = top.merge(
        stations[
            ["station_id", "city_clean"]
        ],
        on="station_id",
        how="left"
    )

    st.dataframe(
        top.sort_values(
            "tickets",
            ascending=False
        ).head(15).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"{unassigned_tickets:,} tickets have no station ID and cannot "
        "be assigned to a selected station city. CSAT is based only on "
        "submitted scores; missing CSAT values are not imputed."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown(
    '<div class="small-muted">'
    'VOLTSCOPE ANALYTICS · Hackathon prototype · '
    'Descriptive insights, not causal claims'
    '</div>',
    unsafe_allow_html=True
)
