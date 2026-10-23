"""
Real Exchange Rate Explorer
---------------------------
Streamlit app that pulls monthly CPI and exchange-rate data from the IMF SDMX 3.0 API
and computes the bilateral real exchange rate between any two IMF countries (or the Euro area).

Definitions
    S(t)   = units of Country 2 currency per 1 unit of Country 1 currency
           = ER2(t) / ER1(t), where ERi(t) is country i's local currency per USD (IMF XDC_USD)
    RER(t) = S(t) * P1(t) / P2(t)
    RER*   = average of RER(t) over valid months in the selected sample
    Real Index(t) = 100 * RER(t) / RER*
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMF_BASE = "https://api.imf.org/external/sdmx/3.0"
START_PERIOD = "1976-01"
EURO_AREA = "G163"
CPI_KEY = "{iso}.CPI._T.IX.M"
HICP_KEY = "{iso}.HICP._T.IX.M"  # Euro area publishes a harmonised index instead of CPI
FX_KEY = "{iso}.XDC_USD.PA_RT.M"

COLOR_REAL = "#4F46E5"
COLOR_NOMINAL = "#0D9488"
COLOR_PPP = "#F59E0B"
COLOR_OVER = "rgba(16, 185, 129, 0.22)"
COLOR_UNDER = "rgba(239, 68, 68, 0.22)"

# ISO3 -> (currency code, ISO2 for flag emoji). Used only for nicer labels.
CURRENCY_INFO: dict[str, tuple[str, str]] = {
    "USA": ("USD", "US"), "GBR": ("GBP", "GB"), "G163": ("EUR", "EU"), "JPN": ("JPY", "JP"),
    "CHN": ("CNY", "CN"), "CAN": ("CAD", "CA"), "AUS": ("AUD", "AU"), "NZL": ("NZD", "NZ"),
    "CHE": ("CHF", "CH"), "SWE": ("SEK", "SE"), "NOR": ("NOK", "NO"), "DNK": ("DKK", "DK"),
    "ISL": ("ISK", "IS"), "IND": ("INR", "IN"), "BRA": ("BRL", "BR"), "MEX": ("MXN", "MX"),
    "ARG": ("ARS", "AR"), "CHL": ("CLP", "CL"), "COL": ("COP", "CO"), "PER": ("PEN", "PE"),
    "URY": ("UYU", "UY"), "ZAF": ("ZAR", "ZA"), "NGA": ("NGN", "NG"), "EGY": ("EGP", "EG"),
    "KEN": ("KES", "KE"), "MAR": ("MAD", "MA"), "TUR": ("TRY", "TR"), "ISR": ("ILS", "IL"),
    "SAU": ("SAR", "SA"), "ARE": ("AED", "AE"), "QAT": ("QAR", "QA"), "KWT": ("KWD", "KW"),
    "RUS": ("RUB", "RU"), "UKR": ("UAH", "UA"), "POL": ("PLN", "PL"), "CZE": ("CZK", "CZ"),
    "HUN": ("HUF", "HU"), "ROU": ("RON", "RO"), "BGR": ("BGN", "BG"), "KOR": ("KRW", "KR"),
    "TWN": ("TWD", "TW"), "HKG": ("HKD", "HK"), "SGP": ("SGD", "SG"), "MYS": ("MYR", "MY"),
    "THA": ("THB", "TH"), "IDN": ("IDR", "ID"), "PHL": ("PHP", "PH"), "VNM": ("VND", "VN"),
    "PAK": ("PKR", "PK"), "BGD": ("BDT", "BD"), "LKA": ("LKR", "LK"), "KAZ": ("KZT", "KZ"),
    "DEU": ("DEM", "DE"), "FRA": ("FRF", "FR"), "ITA": ("ITL", "IT"), "ESP": ("ESP", "ES"),
    "NLD": ("NLG", "NL"), "BEL": ("BEF", "BE"), "AUT": ("ATS", "AT"), "PRT": ("PTE", "PT"),
    "IRL": ("IEP", "IE"), "FIN": ("FIM", "FI"), "GRC": ("GRD", "GR"),
}

# Offline fallback list if the IMF code list cannot be reached.
FALLBACK_NAMES = {
    "USA": "United States", "GBR": "United Kingdom", "G163": "Euro Area (EA)", "JPN": "Japan",
    "CHN": "China, People's Republic of", "CAN": "Canada", "AUS": "Australia", "CHE": "Switzerland",
    "SWE": "Sweden", "NOR": "Norway", "IND": "India", "BRA": "Brazil", "MEX": "Mexico",
    "ZAF": "South Africa", "KOR": "Korea, Republic of", "TUR": "Türkiye, Republic of",
    "DEU": "Germany", "FRA": "France", "ITA": "Italy", "NZL": "New Zealand",
}

PRESETS = {
    "🇬🇧 UK → 🇺🇸 US": ("GBR", "USA"),
    "🇪🇺 Euro → 🇺🇸 US": (EURO_AREA, "USA"),
    "🇺🇸 US → 🇯🇵 Japan": ("USA", "JPN"),
    "🇺🇸 US → 🇨🇳 China": ("USA", "CHN"),
    "🇨🇭 Swiss → 🇪🇺 Euro": ("CHE", EURO_AREA),
    "🇺🇸 US → 🇲🇽 Mexico": ("USA", "MEX"),
}

st.set_page_config(
    page_title="Real Exchange Rate Explorer",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        border-radius: 18px; padding: 1.4rem 1.7rem; margin-bottom: 1.2rem; color: #fff;
        background: linear-gradient(120deg, #312E81 0%, #4F46E5 45%, #0D9488 100%);
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {color: #fff; margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em;}
    .hero p {color: rgba(255,255,255,0.88); margin: 0.35rem 0 0 0; font-size: 1.02rem;}
    .hero .pair {font-size: 1.15rem; font-weight: 600; margin-top: 0.7rem; color: #fff;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22); border-radius: 14px;
        padding: 0.8rem 1rem; background: rgba(128,128,128,0.06);
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
    .verdict {
        border-left: 5px solid var(--vc); border-radius: 10px; padding: 0.8rem 1rem;
        background: rgba(128,128,128,0.07); margin: 0.6rem 0 1rem 0; font-size: 1.02rem;
    }
    .small-note {font-size: 0.85rem; opacity: 0.75;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
    .stTabs [data-baseweb="tab"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# IMF data access
# ---------------------------------------------------------------------------
@st.cache_resource
def _http() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json", "User-Agent": "rer-streamlit-app/1.0"})
    return session


def data_url(flow: str, key: str) -> str:
    return f"{IMF_BASE}/data/dataflow/IMF.STA/{flow}/+/{key}"


def _get_json(url: str, params: dict | None = None) -> dict | None:
    """GET a JSON document. Returns None when the IMF reports 'no data'."""
    resp = _http().get(url, params=params, timeout=60)
    if resp.status_code in (204, 404) or not resp.content:
        return None
    resp.raise_for_status()
    return resp.json()


def _parse_single_series(payload: dict | None) -> pd.Series:
    """Turn an SDMX-JSON 2.0 data message with one series into a monthly pd.Series."""
    if not payload:
        return pd.Series(dtype="float64")
    data = payload.get("data", {})
    datasets = data.get("dataSets", [])
    if not datasets or not datasets[0].get("series"):
        return pd.Series(dtype="float64")
    periods = data["structures"][0]["dimensions"]["observation"][0]["values"]
    series = next(iter(datasets[0]["series"].values()))
    out = {}
    for idx, obs in series.get("observations", {}).items():
        try:
            value = float(obs[0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isnan(value):
            continue
        label = periods[int(idx)]["value"]  # e.g. "1976-M01"
        out[pd.Timestamp(label.replace("-M", "-") + "-01")] = value
    return pd.Series(out, dtype="float64").sort_index()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_series(flow: str, key: str) -> pd.Series:
    params = {"c[TIME_PERIOD]": f"ge:{START_PERIOD}"}
    return _parse_single_series(_get_json(data_url(flow, key), params))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fx(iso: str) -> pd.Series:
    return fetch_series("ER", FX_KEY.format(iso=iso))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_prices(iso: str) -> tuple[pd.Series, str]:
    """Headline CPI; fall back to HICP (used by the Euro area)."""
    for label, template in (("CPI", CPI_KEY), ("HICP", HICP_KEY)):
        s = fetch_series("CPI", template.format(iso=iso))
        if not s.empty:
            return s, label
    return pd.Series(dtype="float64"), "CPI"


def _series_keys(flow: str, key: str) -> set[str]:
    payload = _get_json(data_url(flow, key), {"attributes": "none", "measures": "none"})
    if not payload:
        return set()
    dims = payload["data"]["structures"][0]["dimensions"]["series"]
    codes = dims[0]["values"]
    keys = payload["data"]["dataSets"][0].get("series", {})
    return {codes[int(k.split(":")[0])]["id"] for k in keys}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def country_catalog() -> tuple[dict[str, str], bool]:
    """Countries that have both an IMF exchange-rate series and a CPI/HICP series."""
    try:
        cl = _get_json(f"{IMF_BASE}/structure/codelist/IMF/CL_COUNTRY/+")
        names = {c["id"]: c["name"] for c in cl["data"]["codelists"][0]["codes"]}
        fx = _series_keys("ER", "*.XDC_USD.PA_RT.M")
        prices = _series_keys("CPI", "*.CPI._T.IX.M") | _series_keys("CPI", "*.HICP._T.IX.M")
        available = (fx & prices) | {EURO_AREA}
        catalog = {iso: names.get(iso, iso) for iso in available if iso in names or iso == EURO_AREA}
        catalog[EURO_AREA] = "Euro Area (EA)"
        if len(catalog) < 10:
            raise ValueError("IMF catalog looks incomplete")
        return dict(sorted(catalog.items(), key=lambda kv: kv[1])), True
    except Exception:
        return dict(sorted(FALLBACK_NAMES.items(), key=lambda kv: kv[1])), False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flag(iso: str) -> str:
    info = CURRENCY_INFO.get(iso)
    if not info:
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(ch) - ord("A")) for ch in info[1])


def ccy(iso: str, names: dict[str, str]) -> str:
    info = CURRENCY_INFO.get(iso)
    return info[0] if info else f"{iso} LCU"


def fmt_month(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


def fmt_num(x: float, sig: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:,.{sig}g}" if abs(x) < 1 else f"{x:,.4f}"


def half_life_months(log_rer: pd.Series) -> tuple[float, float]:
    """AR(1) persistence of log RER over consecutive valid months -> (rho, half-life)."""
    q = log_rer.dropna()
    if len(q) < 24:
        return float("nan"), float("nan")
    q = q - q.mean()
    prev = q.shift(1)
    consecutive = (q.index.to_series().diff().dt.days.between(27, 32)).values
    mask = consecutive & prev.notna().values
    x, y = prev.values[mask], q.values[mask]
    if len(x) < 24 or np.allclose(x, 0):
        return float("nan"), float("nan")
    rho = float(np.dot(x, y) / np.dot(x, x))
    hl = math.log(0.5) / math.log(rho) if 0 < rho < 1 else float("inf")
    return rho, hl


# ---------------------------------------------------------------------------
# Session state, URL sharing and callbacks
# ---------------------------------------------------------------------------
names, live_catalog = country_catalog()
isos = list(names.keys())

if "initialized" not in st.session_state:
    qp = st.query_params
    st.session_state.c1 = qp.get("c1", "GBR") if qp.get("c1", "GBR") in names else isos[0]
    st.session_state.c2 = qp.get("c2", "USA") if qp.get("c2", "USA") in names else isos[1]
    st.session_state.view = qp.get("view", "Both") if qp.get("view") in ("Real", "Nominal", "Both") else "Both"
    st.session_state.pending_range = (qp.get("start"), qp.get("end"))
    st.session_state.initialized = True


def swap_countries() -> None:
    st.session_state.c1, st.session_state.c2 = st.session_state.c2, st.session_state.c1


def apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice and choice in PRESETS:
        c1, c2 = PRESETS[choice]
        if c1 in names and c2 in names:
            st.session_state.c1, st.session_state.c2 = c1, c2
    st.session_state.preset = None


def set_range(kind: str) -> None:
    months = st.session_state.get("_months", [])
    if not months:
        return
    last = months[-1]
    if kind == "full":
        start = months[0]
    elif kind == "euro":
        start = max(months[0], "1999-01")
    else:
        years = int(kind)
        start = max(months[0], f"{int(last[:4]) - years}-{last[5:]}")
    start = next((m for m in months if m >= start), months[0])
    st.session_state.range = (start, last)


label_for = lambda iso: f"{flag(iso)}  {names[iso]}  ({ccy(iso, names)})"  # noqa: E731

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💱 Settings")
    st.selectbox("Country 1 (base currency)", isos, key="c1", format_func=label_for)
    st.button("⇅  Swap countries", on_click=swap_countries, width="stretch")
    st.selectbox("Country 2 (quote currency)", isos, key="c2", format_func=label_for)
    st.pills("Quick pairs", list(PRESETS.keys()), key="preset", on_change=apply_preset)
    if not live_catalog:
        st.warning("Couldn't load the full IMF country list. Showing a short offline list.")
    st.divider()
    st.markdown("### Chart options")
    show_ppp = st.toggle("Show PPP fair-value line on nominal", value=True,
                         help="The exchange rate that would put the real index at exactly 100: S = RER* × P2 / P1.")
    shade = st.toggle("Shade over/undervaluation", value=True)
    log_nominal = st.toggle("Log scale for nominal S(t)", value=False)
    st.divider()
    st.caption(
        "Data: IMF SDMX 3.0 API: Exchange Rates (ER) and Consumer Price Index (CPI) datasets. "
        "Monthly from Jan 1976 where available."
    )

c1, c2 = st.session_state.c1, st.session_state.c2
ccy1, ccy2 = ccy(c1, names), ccy(c2, names)

st.markdown(
    f"""
    <div class="hero">
      <h1>Real Exchange Rate Explorer</h1>
      <p>How much do {names[c1]}'s goods cost in terms of {names[c2]}'s goods, relative to the sample average?</p>
      <div class="pair">{flag(c1)} {names[c1]} &nbsp;→&nbsp; {flag(c2)} {names[c2]}
      &nbsp;·&nbsp; S(t) = {ccy2} per 1 {ccy1}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if c1 == c2:
    st.info("Pick two different countries to compute a bilateral real exchange rate.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    with st.spinner(f"Fetching IMF data for {names[c1]} and {names[c2]}…"):
        er1, er2 = fetch_fx(c1), fetch_fx(c2)
        (p1, p1_type), (p2, p2_type) = fetch_prices(c1), fetch_prices(c2)
except requests.RequestException as exc:
    st.error(f"Could not reach the IMF API. Please try again in a minute.\n\n`{exc}`")
    st.stop()

missing = [
    msg for ok, msg in (
        (not er1.empty, f"exchange rate for {names[c1]}"),
        (not er2.empty, f"exchange rate for {names[c2]}"),
        (not p1.empty, f"CPI for {names[c1]}"),
        (not p2.empty, f"CPI for {names[c2]}"),
    ) if not ok
]
if missing:
    st.error("The IMF returned no data for: " + ", ".join(missing) + ". Try another pair.")
    st.stop()

raw = pd.concat({"ER1_LCU_per_USD": er1, "ER2_LCU_per_USD": er2, "P1": p1, "P2": p2}, axis=1)
raw = raw[raw.index >= pd.Timestamp(START_PERIOD + "-01")].sort_index()
raw["S"] = raw["ER2_LCU_per_USD"] / raw["ER1_LCU_per_USD"]
raw["RER"] = raw["S"] * raw["P1"] / raw["P2"]
valid_all = raw["RER"].dropna()

if len(valid_all) < 2:
    st.error("These two countries don't have overlapping monthly CPI and exchange-rate data. Try another pair.")
    st.stop()

# ---------------------------------------------------------------------------
# Sample period selector
# ---------------------------------------------------------------------------
months = [d.strftime("%Y-%m") for d in pd.date_range(valid_all.index[0], valid_all.index[-1], freq="MS")]
st.session_state._months = months

pending = st.session_state.pop("pending_range", None)
if pending and pending[0] in months and pending[1] in months and pending[0] < pending[1]:
    st.session_state.range = pending
current = st.session_state.get("range")
if not current or current[0] not in months or current[1] not in months or current[0] >= current[1]:
    st.session_state.range = (months[0], months[-1])

top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
with top_l:
    st.select_slider(
        "Sample period (RER* is re-computed whenever this changes)",
        options=months,
        key="range",
        format_func=lambda m: pd.Timestamp(m + "-01").strftime("%b %Y"),
    )
with top_r:
    b = st.columns(5)
    b[0].button("Full", on_click=set_range, args=("full",), width="stretch")
    b[1].button("Euro era", on_click=set_range, args=("euro",), width="stretch",
                help="From January 1999")
    b[2].button("20y", on_click=set_range, args=("20",), width="stretch")
    b[3].button("10y", on_click=set_range, args=("10",), width="stretch")
    b[4].button("5y", on_click=set_range, args=("5",), width="stretch")

start_m, end_m = st.session_state.range
start_ts, end_ts = pd.Timestamp(start_m + "-01"), pd.Timestamp(end_m + "-01")

# Keep the URL in sync so any view can be shared by copying the address bar.
st.query_params.update({"c1": c1, "c2": c2, "start": start_m, "end": end_m,
                        "view": st.session_state.get("view") or "Both"})

# ---------------------------------------------------------------------------
# Core calculations on the selected sample
# ---------------------------------------------------------------------------
sample = raw.loc[start_ts:end_ts].copy()
valid = sample["RER"].notna()
n_valid = int(valid.sum())
if n_valid < 2:
    st.warning("Fewer than two valid months in this sample. Widen the date range.")
    st.stop()

rer_star = float(sample.loc[valid, "RER"].mean())
sample["RER_star"] = rer_star
sample["Real_Index"] = 100 * sample["RER"] / rer_star
sample["PPP_FairValue_S"] = rer_star * sample["P2"] / sample["P1"]
sample["Misalignment_pct"] = sample["Real_Index"] - 100
sample["Valid_month"] = valid

v = sample[valid]
last = v.iloc[-1]
last_date = v.index[-1]
rho, hl = half_life_months(np.log(sample["RER"]))
dlog_real = np.log(v["RER"]).diff()
dlog_nom = np.log(v["S"]).diff()
gap_ok = v.index.to_series().diff().dt.days.between(27, 32)
dlog_real, dlog_nom = dlog_real[gap_ok], dlog_nom[gap_ok]
vol_real = float(dlog_real.std() * math.sqrt(12) * 100)
vol_nom = float(dlog_nom.std() * math.sqrt(12) * 100)
corr_rn = float(dlog_real.corr(dlog_nom))

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
m = st.columns(4)
m[0].metric(f"Nominal S · {fmt_month(last_date)}", f"{fmt_num(last['S'])} {ccy2}",
            help=f"{ccy2} per 1 {ccy1}, as reported by the IMF (via USD cross rates).")
m[1].metric("Real index (avg = 100)", f"{last['Real_Index']:.1f}",
            delta=f"{last['Misalignment_pct']:+.1f}% vs average", delta_color="off")
m[2].metric("PPP fair-value S", f"{fmt_num(last['PPP_FairValue_S'])} {ccy2}",
            help="The nominal rate that would make today's real index equal 100: RER* × P2 / P1.")
m[3].metric("Half-life of deviations",
            f"{hl / 12:.1f} yrs" if np.isfinite(hl) else "No mean reversion",
            help="From an AR(1) on log RER over the sample: half-life = ln(0.5) / ln(ρ). "
                 "Classic estimates in the PPP literature are 3–5 years (Rogoff's 'PPP puzzle').")

mis = last["Misalignment_pct"]
if abs(mis) < 2:
    verdict, vc = f"close to its sample-average real value ({mis:+.1f}%)", "#64748B"
elif mis > 0:
    verdict, vc = f"<b>{mis:.1f}% overvalued</b> in real terms vs its sample average", "#10B981"
else:
    verdict, vc = f"<b>{abs(mis):.1f}% undervalued</b> in real terms vs its sample average", "#EF4444"
st.markdown(
    f"""<div class="verdict" style="--vc:{vc}">
    In {fmt_month(last_date)}, the {ccy1} is {verdict} against the {ccy2}.
    At PPP-consistent prices, 1 {ccy1} would buy about <b>{fmt_num(last['PPP_FairValue_S'])} {ccy2}</b>
    instead of {fmt_num(last['S'])}.
    <span class="small-note">&nbsp;Sample {fmt_month(start_ts)}–{fmt_month(end_ts)}, {n_valid} valid months.</span>
    </div>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def real_traces(fig: go.Figure, secondary: bool | None = None) -> None:
    kw = {} if secondary is None else {"secondary_y": secondary}
    x, y = sample.index, sample["Real_Index"]
    if shade:
        base = pd.Series(100.0, index=x)
        above, below = y.clip(lower=100), y.clip(upper=100)
        fig.add_trace(go.Scatter(x=x, y=base, line=dict(width=0), hoverinfo="skip", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=x, y=above, fill="tonexty", fillcolor=COLOR_OVER, line=dict(width=0),
                                 hoverinfo="skip", name=f"{ccy1} overvalued", showlegend=True), **kw)
        fig.add_trace(go.Scatter(x=x, y=base, line=dict(width=0), hoverinfo="skip", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=x, y=below, fill="tonexty", fillcolor=COLOR_UNDER, line=dict(width=0),
                                 hoverinfo="skip", name=f"{ccy1} undervalued", showlegend=True), **kw)
    fig.add_trace(go.Scatter(x=x, y=y, name="Real index (avg = 100)", line=dict(color=COLOR_REAL, width=2.4),
                             hovertemplate="%{y:.2f}<extra>Real index</extra>"), **kw)


def nominal_traces(fig: go.Figure, secondary: bool | None = None) -> None:
    kw = {} if secondary is None else {"secondary_y": secondary}
    fig.add_trace(go.Scatter(x=sample.index, y=sample["S"], name=f"Nominal S ({ccy2} per {ccy1})",
                             line=dict(color=COLOR_NOMINAL, width=2.2),
                             hovertemplate="%{y:,.4f}<extra>Nominal S</extra>"), **kw)
    if show_ppp:
        fig.add_trace(go.Scatter(x=sample.index, y=sample["PPP_FairValue_S"], name="PPP fair value",
                                 line=dict(color=COLOR_PPP, width=1.8, dash="dash"),
                                 hovertemplate="%{y:,.4f}<extra>PPP fair value</extra>"), **kw)


def style(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06), showgrid=False,
        rangeselector=dict(buttons=[
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(count=20, label="20y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )
    return fig


tab_chart, tab_verify, tab_analytics, tab_data, tab_method = st.tabs(
    ["📈 Chart", "🧮 Verify a month", "🔬 Analytics", "📋 Data & CSV", "📖 Method"]
)

with tab_chart:
    view = st.segmented_control("Series", ["Real", "Nominal", "Both"], key="view",
                                label_visibility="collapsed") or "Both"
    if view == "Real":
        fig = go.Figure()
        real_traces(fig)
        fig.add_hline(y=100, line=dict(color="gray", dash="dot", width=1))
        fig.update_yaxes(title_text="Real index (sample avg = 100)")
    elif view == "Nominal":
        fig = go.Figure()
        nominal_traces(fig)
        fig.update_yaxes(title_text=f"{ccy2} per 1 {ccy1}", type="log" if log_nominal else "linear")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        real_traces(fig, secondary=False)
        nominal_traces(fig, secondary=True)
        fig.add_hline(y=100, line=dict(color="gray", dash="dot", width=1), secondary_y=False)
        fig.update_yaxes(title_text="Real index (avg = 100)", color=COLOR_REAL, secondary_y=False)
        fig.update_yaxes(title_text=f"Nominal S: {ccy2} per 1 {ccy1}", color=COLOR_NOMINAL,
                         type="log" if log_nominal else "linear", secondary_y=True, showgrid=False)
    st.plotly_chart(style(fig), width="stretch")
    st.caption(
        f"Real index > 100: {names[c1]}'s goods are relatively expensive (real appreciation of the {ccy1}). "
        f"Nominal S is in actual currency units and is never normalised. "
        f"Prices: {p1_type} for {names[c1]}, {p2_type} for {names[c2]}."
    )

with tab_verify:
    st.markdown("#### Step-by-step RER calculation")
    st.caption("Pick any month to reproduce the numbers in the CSV by hand. Handy for the manual verification.")
    pick = st.select_slider("Month", options=list(v.index), value=v.index[-1], format_func=fmt_month)
    r = v.loc[pick]
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Inputs from the IMF**")
        st.table(pd.DataFrame({
            "Series": [f"ER1: {ccy1} per USD", f"ER2: {ccy2} per USD", f"P1: {names[c1]} {p1_type}",
                       f"P2: {names[c2]} {p2_type}", "RER* (sample average)"],
            "Value": [f"{r['ER1_LCU_per_USD']:.6f}", f"{r['ER2_LCU_per_USD']:.6f}", f"{r['P1']:.4f}",
                      f"{r['P2']:.4f}", f"{rer_star:.6f}"],
        }).set_index("Series"))
    with cB:
        st.markdown("**Calculation**")
        st.latex(rf"S = \frac{{ER_2}}{{ER_1}} = \frac{{{r['ER2_LCU_per_USD']:.6f}}}{{{r['ER1_LCU_per_USD']:.6f}}}"
                 rf"= {r['S']:.6f}\ \text{{{ccy2} per {ccy1}}}")
        st.latex(rf"RER = S \times \frac{{P_1}}{{P_2}} = {r['S']:.6f} \times \frac{{{r['P1']:.4f}}}{{{r['P2']:.4f}}}"
                 rf"= {r['RER']:.6f}")
        st.latex(rf"\text{{Real Index}} = 100 \times \frac{{RER}}{{RER^*}} = 100 \times "
                 rf"\frac{{{r['RER']:.6f}}}{{{rer_star:.6f}}} = {r['Real_Index']:.4f}")
    avg_idx = v["Real_Index"].mean()
    st.success(f"✅ Check: the average Real Index over the {n_valid} valid months in this sample is "
               f"**{avg_idx:.10f}** (should equal 100).")

with tab_analytics:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Real volatility (ann.)", f"{vol_real:.1f}%", help="Std. dev. of monthly Δlog RER × √12.")
    a2.metric("Nominal volatility (ann.)", f"{vol_nom:.1f}%", help="Std. dev. of monthly Δlog S × √12.")
    a3.metric("Corr(ΔReal, ΔNominal)", f"{corr_rn:.2f}",
              help="Close to 1 under floating rates: nominal moves drive real moves because prices are sticky (Mussa, 1986).")
    a4.metric("AR(1) ρ of log RER", f"{rho:.3f}" if np.isfinite(rho) else "—")

    st.markdown("#### Where did the real change come from?")
    st.caption(
        f"Δln RER = Δln S + Δln P1 − Δln P2, from {fmt_month(v.index[0])} to {fmt_month(v.index[-1])} "
        "(in log points × 100 ≈ %)."
    )
    first = v.iloc[0]
    d_s = 100 * math.log(last["S"] / first["S"])
    d_p1 = 100 * math.log(last["P1"] / first["P1"])
    d_p2 = -100 * math.log(last["P2"] / first["P2"])
    wf = go.Figure(go.Waterfall(
        x=[f"Nominal S ({ccy2}/{ccy1})", f"{names[c1]} prices (+P1)", f"{names[c2]} prices (−P2)", "Real exchange rate"],
        measure=["relative", "relative", "relative", "total"],
        y=[d_s, d_p1, d_p2, 0],
        text=[f"{d_s:+.1f}", f"{d_p1:+.1f}", f"{d_p2:+.1f}", f"{d_s + d_p1 + d_p2:+.1f}"],
        textposition="outside",
        increasing=dict(marker=dict(color="#10B981")), decreasing=dict(marker=dict(color="#EF4444")),
        totals=dict(marker=dict(color=COLOR_REAL)),
    ))
    wf.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                     yaxis_title="log points × 100")
    st.plotly_chart(wf, width="stretch")

    cL, cR = st.columns(2)
    with cL:
        st.markdown("#### Cumulative decomposition over time")
        dec = pd.DataFrame({
            "Nominal S": 100 * np.log(v["S"] / first["S"]),
            "Inflation differential (P1/P2)": 100 * np.log((v["P1"] / v["P2"]) / (first["P1"] / first["P2"])),
            "Real exchange rate": 100 * np.log(v["RER"] / first["RER"]),
        })
        fd = go.Figure()
        for col, color in zip(dec.columns, [COLOR_NOMINAL, COLOR_PPP, COLOR_REAL]):
            fd.add_trace(go.Scatter(x=dec.index, y=dec[col], name=col, line=dict(color=color, width=2)))
        fd.update_layout(height=360, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                         legend=dict(orientation="h", y=-0.15), yaxis_title="log points × 100 since start")
        st.plotly_chart(fd, width="stretch")
    with cR:
        st.markdown("#### Distribution of the real index")
        fh = go.Figure(go.Histogram(x=v["Real_Index"], nbinsx=40, marker_color=COLOR_REAL, opacity=0.85))
        fh.add_vline(x=100, line=dict(color="gray", dash="dot"))
        fh.add_vline(x=float(last["Real_Index"]), line=dict(color=COLOR_PPP, width=2),
                     annotation_text="latest", annotation_position="top")
        fh.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), bargap=0.05,
                         xaxis_title="Real index", yaxis_title="Months")
        st.plotly_chart(fh, width="stretch")
        pct = float((v["Real_Index"] < last["Real_Index"]).mean() * 100)
        st.caption(f"The latest reading is higher than {pct:.0f}% of months in the sample.")

    st.markdown("#### Extremes in the sample")
    hi, lo = v["Real_Index"].idxmax(), v["Real_Index"].idxmin()
    e1, e2, e3 = st.columns(3)
    e1.metric("Strongest real {0}".format(ccy1), f"{v.loc[hi, 'Real_Index']:.1f}", fmt_month(hi), delta_color="off")
    e2.metric("Weakest real {0}".format(ccy1), f"{v.loc[lo, 'Real_Index']:.1f}", fmt_month(lo), delta_color="off")
    e3.metric("Range of nominal S", f"{v['S'].min():,.4g} – {v['S'].max():,.4g}", f"{ccy2} per {ccy1}",
              delta_color="off")

with tab_data:
    export = sample.reset_index(names="Date")
    export["Date"] = export["Date"].dt.strftime("%Y-%m")
    export.insert(1, "Country1", f"{names[c1]} ({c1})")
    export.insert(2, "Country2", f"{names[c2]} ({c2})")
    export = export[[
        "Date", "Country1", "Country2", "ER1_LCU_per_USD", "ER2_LCU_per_USD", "S", "P1", "P2",
        "RER", "RER_star", "Real_Index", "PPP_FairValue_S", "Misalignment_pct", "Valid_month",
    ]]
    gaps = len(sample) - n_valid
    d1, d2, d3 = st.columns(3)
    d1.metric("Months in sample", len(sample))
    d2.metric("Valid months", n_valid)
    d3.metric("Months with gaps", gaps, help="A month is valid only if S, P1 and P2 are all available.")
    st.download_button(
        "⬇️  Download CSV (all observations and calculations)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"rer_{c1}_{c2}_{start_m}_{end_m}.csv",
        mime="text/csv",
        type="primary",
        width="stretch",
    )
    st.dataframe(
        export, hide_index=True, width="stretch", height=460,
        column_config={
            "ER1_LCU_per_USD": st.column_config.NumberColumn(f"ER1 ({ccy1}/USD)", format="%.6f"),
            "ER2_LCU_per_USD": st.column_config.NumberColumn(f"ER2 ({ccy2}/USD)", format="%.6f"),
            "S": st.column_config.NumberColumn(f"S ({ccy2}/{ccy1})", format="%.6f"),
            "P1": st.column_config.NumberColumn(f"P1 {p1_type}", format="%.4f"),
            "P2": st.column_config.NumberColumn(f"P2 {p2_type}", format="%.4f"),
            "RER": st.column_config.NumberColumn(format="%.6f"),
            "RER_star": st.column_config.NumberColumn("RER*", format="%.6f"),
            "Real_Index": st.column_config.ProgressColumn(
                "Real index", format="%.2f",
                min_value=float(v["Real_Index"].min()), max_value=float(v["Real_Index"].max())),
            "PPP_FairValue_S": st.column_config.NumberColumn("PPP fair S", format="%.6f"),
            "Misalignment_pct": st.column_config.NumberColumn("Misalign. %", format="%+.2f"),
        },
    )
    with st.expander("Data coverage for this pair"):
        cov = pd.DataFrame({
            "Series": [f"ER {names[c1]}", f"ER {names[c2]}", f"{p1_type} {names[c1]}", f"{p2_type} {names[c2]}"],
            "First month": [fmt_month(s.index[0]) for s in (er1, er2, p1, p2)],
            "Last month": [fmt_month(s.index[-1]) for s in (er1, er2, p1, p2)],
            "Observations": [len(s) for s in (er1, er2, p1, p2)],
        })
        st.dataframe(cov, hide_index=True, width="stretch")

with tab_method:
    st.markdown(
        f"""
**Exchange rate.** The IMF reports each currency in local currency units per US dollar (`XDC_USD`).
The bilateral rate is the cross rate

$$S(t) = \\frac{{ER_2(t)}}{{ER_1(t)}} = \\frac{{\\text{{{ccy2} per USD}}}}{{\\text{{{ccy1} per USD}}}} = \\text{{{ccy2} per 1 {ccy1}}}$$

**Real exchange rate.** With $P_1$ and $P_2$ the monthly consumer price indices,

$$RER(t) = S(t)\\,\\frac{{P_1(t)}}{{P_2(t)}}$$

It measures the price of {names[c1]}'s consumption basket in units of {names[c2]}'s basket. Because CPIs
are indices with arbitrary base years, the *level* of RER has no meaning, so we normalise it:

$$RER^* = \\frac{{1}}{{N}}\\sum_{{t \\in \\text{{sample}}}} RER(t), \\qquad \\text{{Real Index}}(t) = 100\\,\\frac{{RER(t)}}{{RER^*}}$$

where the average runs over the $N$ months in the selected sample with all three inputs available.
$RER^*$ is recomputed whenever the sample changes, so the index always averages exactly 100.

**PPP fair value.** The nominal rate that would put the real index at 100 given current prices:
$S^{{PPP}}(t) = RER^* \\, P_2(t) / P_1(t)$. This is relative PPP anchored to the sample average.

**Half-life.** An AR(1) on demeaned $\\ln RER$, $q_t = \\rho\\, q_{{t-1}} + \\varepsilon_t$, gives the half-life
$\\ln(0.5)/\\ln(\\rho)$ months: how long it takes for half of a deviation from the average to disappear.

**Data sources** (IMF SDMX 3.0, monthly, from {START_PERIOD}):
- Exchange rate: `{data_url("ER", FX_KEY.format(iso=c1))}`
- Exchange rate: `{data_url("ER", FX_KEY.format(iso=c2))}`
- Prices: `{data_url("CPI", (CPI_KEY if p1_type == "CPI" else HICP_KEY).format(iso=c1))}`
- Prices: `{data_url("CPI", (CPI_KEY if p2_type == "CPI" else HICP_KEY).format(iso=c2))}`

**Caveats.** IMF series can have gaps or end early (e.g. pre-euro national currencies stop in 1998, and the
Euro area exchange rate starts in 1999). The Euro area publishes a harmonised index (HICP), which this app uses
automatically when no `CPI` series exists. Data is cached for six hours.
        """
    )

st.markdown(
    f"<div class='small-note' style='text-align:center;margin-top:2rem'>"
    f"Built with Streamlit · Data: International Monetary Fund · "
    f"Loaded {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')} · "
    f"Tip: the URL updates with your selections, so you can copy it to share this exact view.</div>",
    unsafe_allow_html=True,
)
