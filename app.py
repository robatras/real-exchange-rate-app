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
from concurrent.futures import ThreadPoolExecutor
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

# Palette tuned for the dark theme in .streamlit/config.toml
COLOR_REAL = "#818CF8"
COLOR_NOMINAL = "#2DD4BF"
COLOR_PPP = "#FBBF24"
COLOR_UP = "#34D399"
COLOR_DOWN = "#FB7185"
COLOR_OVER = "rgba(52, 211, 153, 0.18)"
COLOR_UNDER = "rgba(251, 113, 133, 0.18)"
MAP_SCALE = [[0, "#FF4D6D"], [0.3, "#9F1239"], [0.5, "#334155"], [0.7, "#047857"], [1, "#4ADE80"]]
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# Current euro members. Their own IMF exchange-rate series stop in the year before they joined,
# so the world map shows them with the Euro area's value.
EURO_MEMBERS = {
    "AUT", "BEL", "CYP", "DEU", "ESP", "EST", "FIN", "FRA", "GRC", "HRV",
    "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD", "PRT", "SVK", "SVN",
}

# ISO3 currency pairs, used only for nicer labels. Euro members show their pre-euro
# currency because their own IMF exchange-rate series end when they joined the euro.
_CURRENCY_TABLE = """
G163 EUR  AFG AFN  ALB ALL  DZA DZD  AND EUR  AGO AOA  AIA XCD  ATG XCD
ARG ARS  ARM AMD  ABW AWG  AUS AUD  AUT ATS  AZE AZN  BHS BSD  BHR BHD
BGD BDT  BRB BBD  BLR BYN  BEL BEF  BLZ BZD  BEN XOF  BTN BTN  BOL BOB
BIH BAM  BWA BWP  BRA BRL  BRN BND  BGR BGN  BFA XOF  BDI BIF  CPV CVE
KHM KHR  CMR XAF  CAN CAD  CAF XAF  TCD XAF  CHL CLP  CHN CNY  COL COP
COM KMF  COD CDF  COG XAF  CRI CRC  CIV XOF  HRV HRK  CUW ANG  CYP CYP
CZE CZK  DNK DKK  DJI DJF  DMA XCD  DOM DOP  ECU USD  EGY EGP  SLV USD
GNQ XAF  ERI ERN  EST EEK  SWZ SZL  ETH ETB  FJI FJD  FIN FIM  FRA FRF
GAB XAF  GMB GMD  GEO GEL  DEU DEM  GHA GHS  GRC GRD  GRD XCD  GTM GTQ
GIN GNF  GNB XOF  GUY GYD  HTI HTG  HND HNL  HKG HKD  HUN HUF  ISL ISK
IND INR  IDN IDR  IRN IRR  IRQ IQD  IRL IEP  ISR ILS  ITA ITL  JAM JMD
JPN JPY  JOR JOD  KAZ KZT  KEN KES  KIR AUD  KOR KRW  KWT KWD  KGZ KGS
LAO LAK  LVA LVL  LBN LBP  LSO LSL  LBR LRD  LBY LYD  LTU LTL  LUX LUF
MAC MOP  MDG MGA  MWI MWK  MYS MYR  MDV MVR  MLI XOF  MLT MTL  MHL USD
MRT MRU  MUS MUR  MEX MXN  FSM USD  MDA MDL  MNG MNT  MNE EUR  MSR XCD
MAR MAD  MOZ MZN  MMR MMK  NAM NAD  NRU AUD  NPL NPR  NLD NLG  NZL NZD
NIC NIO  NER XOF  NGA NGN  MKD MKD  NOR NOK  OMN OMR  PAK PKR  PLW USD
PAN PAB  PNG PGK  PRY PYG  PER PEN  PHL PHP  POL PLN  PRT PTE  QAT QAR
ROU RON  RUS RUB  RWA RWF  KNA XCD  LCA XCD  VCT XCD  WSM WST  SMR EUR
STP STN  SAU SAR  SEN XOF  SRB RSD  SYC SCR  SLE SLE  SGP SGD  SVK SKK
SVN SIT  SLB SBD  SOM SOS  ZAF ZAR  SSD SSP  ESP ESP  LKA LKR  SDN SDG
SUR SRD  SWE SEK  CHE CHF  SYR SYP  TWN TWD  TJK TJS  TZA TZS  THA THB
TLS USD  TGO XOF  TON TOP  TTO TTD  TUN TND  TUR TRY  TKM TMT  TUV AUD
UGA UGX  UKR UAH  ARE AED  GBR GBP  USA USD  URY UYU  UZB UZS  VUT VUV
VEN VES  VNM VND  YEM YER  ZMB ZMW  ZWE ZWG  PRI USD  KOS EUR  WBG ILS
CUB CUP
"""
_tokens = _CURRENCY_TABLE.split()
CURRENCY_CODES: dict[str, str] = dict(zip(_tokens[::2], _tokens[1::2]))

# Offline fallback list if the IMF code list cannot be reached.
FALLBACK_NAMES = {
    "USA": "United States", "GBR": "United Kingdom", "G163": "Euro Area (EA)", "JPN": "Japan",
    "CHN": "China, People's Republic of", "CAN": "Canada", "AUS": "Australia", "CHE": "Switzerland",
    "SWE": "Sweden", "NOR": "Norway", "IND": "India", "BRA": "Brazil", "MEX": "Mexico",
    "ZAF": "South Africa", "KOR": "Korea, Republic of", "TUR": "Türkiye, Republic of",
    "DEU": "Germany", "FRA": "France", "ITA": "Italy", "NZL": "New Zealand",
}

PRESETS = {
    "UK → US": ("GBR", "USA"),
    "Euro → US": (EURO_AREA, "USA"),
    "US → Japan": ("USA", "JPN"),
    "US → China": ("USA", "CHN"),
    "Swiss → Euro": ("CHE", EURO_AREA),
    "US → Mexico": ("USA", "MEX"),
}

SEC_MAP, SEC_CHART = "🌍 World map", "📈 Chart"
SECTIONS = [SEC_MAP, SEC_CHART, "🧮 Verify", "🔬 Analytics", "📋 Data & CSV", "📖 Method"]

st.set_page_config(
    page_title="Real Exchange Rate Explorer",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp button, .stApp input, .hero, .kpi {font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;}
    .stApp {
        background:
            radial-gradient(1100px 560px at 8% -8%, rgba(99,102,241,0.20), transparent 60%),
            radial-gradient(900px 480px at 100% 0%, rgba(20,184,166,0.13), transparent 60%),
            #0B1020;
    }
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(129,140,248,0.14);}
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
    h4 {letter-spacing: -0.01em;}

    /* Hero */
    @keyframes drift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
    .hero {
        position: relative; overflow: hidden; border-radius: 22px; padding: 1.6rem 1.9rem; margin-bottom: 1.1rem;
        color: #fff; background: linear-gradient(115deg, #1E1B4B, #4338CA, #6366F1, #0F766E, #14B8A6);
        background-size: 300% 300%; animation: drift 18s ease infinite;
        box-shadow: 0 18px 40px -12px rgba(67, 56, 202, 0.45);
        display: flex; flex-wrap: wrap; gap: 1.2rem 2.5rem; align-items: center; justify-content: space-between;
    }
    .hero::after {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background: radial-gradient(circle at 85% 0%, rgba(255,255,255,0.18), transparent 45%);
    }
    .hero .eyebrow {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.8;}
    .hero h1 {color: #fff; margin: 0.15rem 0 0 0; padding: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;}
    .hero .sub {color: rgba(255,255,255,0.85); margin-top: 0.4rem; font-size: 1rem; max-width: 40rem;}
    .hero .quote {text-align: right; z-index: 1;}
    .hero .quote .big {font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; font-variant-numeric: tabular-nums;}
    .hero .quote .big small {font-size: 1.1rem; font-weight: 600; opacity: 0.85;}
    .chip {
        display: inline-block; margin-top: 0.5rem; padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.85rem;
        font-weight: 600; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); backdrop-filter: blur(6px);
    }

    /* KPI cards */
    .kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; margin: 0.2rem 0 1rem 0;}
    .kpi {
        border: 1px solid rgba(129,140,248,0.16); border-radius: 16px; padding: 0.95rem 1.1rem 0.7rem 1.1rem;
        background: linear-gradient(160deg, rgba(129,140,248,0.09), rgba(21,27,51,0.6) 55%, rgba(45,212,191,0.05));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .kpi:hover {transform: translateY(-3px); box-shadow: 0 14px 34px -14px rgba(129,140,248,0.6); border-color: rgba(129,140,248,0.5);}
    .kpi .label {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.65;}
    .kpi .value {font-size: 1.65rem; font-weight: 750; letter-spacing: -0.02em; margin-top: 0.15rem; font-variant-numeric: tabular-nums;}
    .kpi .value small {font-size: 0.9rem; font-weight: 600; opacity: 0.6;}
    .kpi .foot {display: flex; justify-content: space-between; align-items: flex-end; gap: 0.5rem; margin-top: 0.25rem;}
    .kpi .note {font-size: 0.8rem; opacity: 0.7;}
    .pos {color: #34D399; font-weight: 700;} .neg {color: #FB7185; font-weight: 700;}

    div[data-testid="stMetric"] {
        border: 1px solid rgba(129,140,248,0.16); border-radius: 16px; padding: 0.8rem 1rem;
        background: linear-gradient(160deg, rgba(129,140,248,0.08), rgba(21,27,51,0.6));
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.75;}

    .verdict {
        border-radius: 14px; padding: 0.85rem 1.1rem; margin: 0 0 1.1rem 0; font-size: 1rem; line-height: 1.55;
        background: linear-gradient(90deg, color-mix(in srgb, var(--vc) 14%, transparent), rgba(128,128,128,0.04));
        border: 1px solid color-mix(in srgb, var(--vc) 35%, transparent);
    }
    .small-note {font-size: 0.84rem; opacity: 0.7;}
    .section-title {font-size: 1.15rem; font-weight: 700; margin: 0.4rem 0 0.2rem 0; letter-spacing: -0.01em;}
    footer {visibility: hidden;}
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
    resp = _http().get(url, params=params, timeout=120)
    if resp.status_code in (204, 404) or not resp.content:
        return None
    resp.raise_for_status()
    return resp.json()


def _period_to_ts(label: str) -> pd.Timestamp:
    return pd.Timestamp(label.replace("-M", "-") + "-01")  # "1976-M01" -> 1976-01-01


def _parse_observations(observations: dict, periods: list) -> pd.Series:
    out = {}
    for idx, obs in observations.items():
        try:
            value = float(obs[0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isnan(value):
            continue
        out[_period_to_ts(periods[int(idx)]["value"])] = value
    return pd.Series(out, dtype="float64").sort_index()


def _parse_panel(payload: dict | None) -> pd.DataFrame:
    """Turn an SDMX-JSON 2.0 data message into a DataFrame: one column per country."""
    if not payload:
        return pd.DataFrame()
    data = payload.get("data", {})
    datasets = data.get("dataSets", [])
    if not datasets or not datasets[0].get("series"):
        return pd.DataFrame()
    structure = data["structures"][0]["dimensions"]
    periods = structure["observation"][0]["values"]
    countries = structure["series"][0]["values"]
    cols = {}
    for key, series in datasets[0]["series"].items():
        iso = countries[int(key.split(":")[0])]["id"]
        cols[iso] = _parse_observations(series.get("observations", {}), periods)
    return pd.DataFrame(cols).sort_index()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_series(flow: str, key: str) -> pd.Series:
    params = {"c[TIME_PERIOD]": f"ge:{START_PERIOD}", "attributes": "none"}
    panel = _parse_panel(_get_json(data_url(flow, key), params))
    return panel.iloc[:, 0].dropna() if not panel.empty else pd.Series(dtype="float64")


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


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fetch_world_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exchange rates and CPI (HICP where no CPI exists) for every country, downloaded in parallel."""
    params = {"c[TIME_PERIOD]": f"ge:{START_PERIOD}", "attributes": "none"}
    queries = [("ER", FX_KEY), ("CPI", CPI_KEY), ("CPI", HICP_KEY)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        payloads = pool.map(lambda fk: _get_json(data_url(fk[0], fk[1].format(iso="*")), params), queries)
        fx, cpi, hicp = (_parse_panel(p) for p in payloads)
    for iso in hicp.columns.difference(cpi.columns):
        cpi[iso] = hicp[iso]
    return fx, cpi


def _series_keys(flow: str, key: str) -> set[str]:
    payload = _get_json(data_url(flow, key), {"attributes": "none", "measures": "none"})
    if not payload:
        return set()
    codes = payload["data"]["structures"][0]["dimensions"]["series"][0]["values"]
    keys = payload["data"]["dataSets"][0].get("series", {})
    return {codes[int(k.split(":")[0])]["id"] for k in keys}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def country_catalog() -> tuple[dict[str, str], bool]:
    """Countries that have both an IMF exchange-rate series and a CPI/HICP series."""
    try:
        cl = _get_json(f"{IMF_BASE}/structure/codelist/IMF/CL_COUNTRY/+")
        names = {c["id"]: c["name"] for c in cl["data"]["codelists"][0]["codes"]}
        fx = _series_keys("ER", FX_KEY.format(iso="*"))
        prices = _series_keys("CPI", CPI_KEY.format(iso="*")) | _series_keys("CPI", HICP_KEY.format(iso="*"))
        catalog = {iso: names[iso] for iso in (fx & prices) if iso in names}
        catalog[EURO_AREA] = "Euro Area (EA)"
        if len(catalog) < 10:
            raise ValueError("IMF catalog looks incomplete")
        return dict(sorted(catalog.items(), key=lambda kv: kv[1])), True
    except Exception:
        return dict(sorted(FALLBACK_NAMES.items(), key=lambda kv: kv[1])), False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ccy(iso: str) -> str:
    return CURRENCY_CODES.get(iso, f"{iso} LCU")


def fmt_month(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


def fmt_num(x: float, sig: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:,.{sig}g}" if abs(x) < 1 else f"{x:,.4f}"


def sparkline(values: pd.Series, color: str, width: int = 120, height: int = 34) -> str:
    """Tiny inline SVG trend line for the KPI cards."""
    y = values.dropna().to_numpy(dtype=float)
    if len(y) < 2:
        return ""
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo or 1.0
    xs = np.linspace(2, width - 2, len(y))
    ys = height - 3 - (y - lo) / span * (height - 6)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    area = f"2,{height} {pts} {width - 2},{height}"
    gid = f"g{abs(hash((color, pts))) % 10**8}"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.8" fill="{color}"/></svg>'
    )


def kpi(label: str, value: str, note: str = "", spark: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="foot"><div class="note">{note}</div>{spark}</div></div>')


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


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(
        height=height, font=dict(family=FONT, size=13), margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Session state, URL sharing and callbacks
# ---------------------------------------------------------------------------
names, live_catalog = country_catalog()
isos = list(names.keys())

if "initialized" not in st.session_state:
    qp = st.query_params
    st.session_state.c1 = qp.get("c1") if qp.get("c1") in names else ("GBR" if "GBR" in names else isos[0])
    st.session_state.c2 = qp.get("c2") if qp.get("c2") in names else ("USA" if "USA" in names else isos[1])
    st.session_state.view = qp.get("view") if qp.get("view") in ("Real", "Nominal", "Both") else "Both"
    st.session_state.section = next((s for s in SECTIONS if s.split(" ", 1)[1] == qp.get("tab")), SECTIONS[0])
    st.session_state.pending_range = (qp.get("start"), qp.get("end"))
    st.session_state.initialized = True

# A click on the world map asks for a new pair; apply it before the widgets are drawn.
if "pending_pair" in st.session_state:
    st.session_state.c1, st.session_state.c2 = st.session_state.pop("pending_pair")
    st.session_state.section = SEC_CHART


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
        start = max(months[0], f"{int(last[:4]) - int(kind)}-{last[5:]}")
    start = next((m for m in months if m >= start), months[0])
    st.session_state.range = (start, last)


def label_for(iso: str) -> str:
    return f"{names[iso]}  ({ccy(iso)})"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💱 Currency pair")
    st.selectbox("Country 1 (base currency)", isos, key="c1", format_func=label_for)
    st.button("⇅  Swap countries", on_click=swap_countries, width="stretch")
    st.selectbox("Country 2 (quote currency)", isos, key="c2", format_func=label_for)
    st.pills("Quick pairs", list(PRESETS.keys()), key="preset", on_change=apply_preset)
    if not live_catalog:
        st.warning("Couldn't load the full IMF country list. Showing a short offline list.")
    st.divider()
    st.markdown("### 🎛️ Chart options")
    show_ppp = st.toggle("PPP fair-value line", value=True,
                         help="The exchange rate that would put the real index at exactly 100: S = RER* × P2 / P1.")
    shade = st.toggle("Shade over/undervaluation", value=True)
    log_nominal = st.toggle("Log scale for nominal S(t)", value=False)
    st.divider()
    st.caption(
        "Data: IMF SDMX 3.0 API: Exchange Rates (ER) and Consumer Price Index (CPI) datasets. "
        "Monthly from Jan 1976 where available."
    )

c1, c2 = st.session_state.c1, st.session_state.c2
ccy1, ccy2 = ccy(c1), ccy(c2)

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

raw = pd.concat({"ER1_LCU_per_USD": er1, "ER2_LCU_per_USD": er2, "P1": p1, "P2": p2}, axis=1, sort=True)
raw = raw[raw.index >= pd.Timestamp(START_PERIOD + "-01")].sort_index()
raw["S"] = raw["ER2_LCU_per_USD"] / raw["ER1_LCU_per_USD"]
raw["RER"] = raw["S"] * raw["P1"] / raw["P2"]
valid_all = raw["RER"].dropna()

if len(valid_all) < 2:
    st.error("These two countries don't have overlapping monthly CPI and exchange-rate data. Try another pair.")
    st.stop()

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
months = [d.strftime("%Y-%m") for d in pd.date_range(valid_all.index[0], valid_all.index[-1], freq="MS")]
st.session_state._months = months

pending = st.session_state.pop("pending_range", None)
if pending and pending[0] in months and pending[1] in months and pending[0] < pending[1]:
    st.session_state.range = pending
current = st.session_state.get("range")
if not current or current[0] not in months or current[1] not in months or current[0] >= current[1]:
    st.session_state.range = (months[0], months[-1])

start_m, end_m = st.session_state.range
start_ts, end_ts = pd.Timestamp(start_m + "-01"), pd.Timestamp(end_m + "-01")

# ---------------------------------------------------------------------------
# Core calculations on the selected sample
# ---------------------------------------------------------------------------
sample = raw.loc[start_ts:end_ts].copy()
valid = sample["RER"].notna()
n_valid = int(valid.sum())

rer_star = float(sample.loc[valid, "RER"].mean())
sample["RER_star"] = rer_star
sample["Real_Index"] = 100 * sample["RER"] / rer_star
sample["PPP_FairValue_S"] = rer_star * sample["P2"] / sample["P1"]
sample["Misalignment_pct"] = sample["Real_Index"] - 100
sample["Valid_month"] = valid

v = sample[valid]
last = v.iloc[-1]
last_date = v.index[-1]
mis = float(last["Misalignment_pct"])
rho, hl = half_life_months(np.log(sample["RER"]))
dlog_real = np.log(v["RER"]).diff()
dlog_nom = np.log(v["S"]).diff()
gap_ok = v.index.to_series().diff().dt.days.between(27, 32)
dlog_real, dlog_nom = dlog_real[gap_ok], dlog_nom[gap_ok]
vol_real = float(dlog_real.std() * math.sqrt(12) * 100)
vol_nom = float(dlog_nom.std() * math.sqrt(12) * 100)
corr_rn = float(dlog_real.corr(dlog_nom))

# ---------------------------------------------------------------------------
# Hero + KPI cards
# ---------------------------------------------------------------------------
chip_txt = (f"Real index {last['Real_Index']:.1f} · {abs(mis):.1f}% "
            f"{'above' if mis >= 0 else 'below'} average")
st.markdown(
    f"""
    <div class="hero">
      <div style="z-index:1">
        <div class="eyebrow">Real Exchange Rate Explorer · IMF data</div>
        <h1>{names[c1]} → {names[c2]}</h1>
        <div class="sub">How expensive are {names[c1]}'s goods in terms of {names[c2]}'s goods,
        relative to the sample average? S(t) = {ccy2} per 1 {ccy1}.</div>
      </div>
      <div class="quote">
        <div class="eyebrow">{fmt_month(last_date)}</div>
        <div class="big">1 {ccy1} = {fmt_num(last['S'])} <small>{ccy2}</small></div>
        <span class="chip">{chip_txt}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

recent = v.loc[last_date - pd.DateOffset(years=5):]
mis_cls = "pos" if mis >= 0 else "neg"
st.markdown(
    '<div class="kpis">'
    + kpi(f"Nominal S · {ccy2} per {ccy1}", f"{fmt_num(last['S'])}",
          "Last 5 years", sparkline(recent["S"], COLOR_NOMINAL))
    + kpi("Real index (avg = 100)", f"{last['Real_Index']:.1f}",
          f'<span class="{mis_cls}">{mis:+.1f}%</span> vs average', sparkline(recent["Real_Index"], COLOR_REAL))
    + kpi("PPP fair-value S", f"{fmt_num(last['PPP_FairValue_S'])} <small>{ccy2}</small>",
          "RER* × P2 / P1", sparkline(recent["PPP_FairValue_S"], COLOR_PPP))
    + kpi("Half-life of deviations",
          f"{hl / 12:.1f} <small>yrs</small>" if np.isfinite(hl) else "∞",
          f"AR(1) ρ = {rho:.3f}" if np.isfinite(rho) else "Too few months")
    + "</div>",
    unsafe_allow_html=True,
)

if abs(mis) < 2:
    verdict, vc = f"close to its sample-average real value ({mis:+.1f}%)", "#64748B"
elif mis > 0:
    verdict, vc = f"<b>{mis:.1f}% overvalued</b> in real terms vs its sample average", COLOR_UP
else:
    verdict, vc = f"<b>{abs(mis):.1f}% undervalued</b> in real terms vs its sample average", COLOR_DOWN
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
# Controls: sample period + section navigation
# ---------------------------------------------------------------------------
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
    b[1].button("Euro era", on_click=set_range, args=("euro",), width="stretch", help="From January 1999")
    b[2].button("20y", on_click=set_range, args=("20",), width="stretch")
    b[3].button("10y", on_click=set_range, args=("10",), width="stretch")
    b[4].button("5y", on_click=set_range, args=("5",), width="stretch")

section = st.segmented_control("Section", SECTIONS, key="section", label_visibility="collapsed") or SECTIONS[0]

# Keep the URL in sync so any view can be shared by copying the address bar.
st.query_params.update({"c1": c1, "c2": c2, "start": start_m, "end": end_m,
                        "view": st.session_state.get("view") or "Both", "tab": section.split(" ", 1)[1]})


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def real_traces(fig: go.Figure, secondary: bool | None = None) -> None:
    kw = {} if secondary is None else {"secondary_y": secondary}
    x, y = sample.index, sample["Real_Index"]
    if shade:
        base = pd.Series(100.0, index=x)
        fig.add_trace(go.Scatter(x=x, y=base, line=dict(width=0), hoverinfo="skip", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=x, y=y.clip(lower=100), fill="tonexty", fillcolor=COLOR_OVER, line=dict(width=0),
                                 hoverinfo="skip", name=f"{ccy1} overvalued"), **kw)
        fig.add_trace(go.Scatter(x=x, y=base, line=dict(width=0), hoverinfo="skip", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=x, y=y.clip(upper=100), fill="tonexty", fillcolor=COLOR_UNDER, line=dict(width=0),
                                 hoverinfo="skip", name=f"{ccy1} undervalued"), **kw)
    fig.add_trace(go.Scatter(x=x, y=y, name="Real index (avg = 100)", line=dict(color=COLOR_REAL, width=2.6),
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


def time_axis(fig: go.Figure) -> go.Figure:
    # No Plotly range buttons: they would only zoom, while the sample buttons above also recompute RER*.
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return fig


def section_chart() -> None:
    view = st.segmented_control("Series", ["Real", "Nominal", "Both"], key="view") or "Both"
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
    st.plotly_chart(time_axis(base_layout(fig, 540)), width="stretch")
    st.caption(
        f"Real index > 100: {names[c1]}'s goods are relatively expensive (real appreciation of the {ccy1}). "
        f"Nominal S is in actual currency units and is never normalised. "
        f"Prices: {p1_type} for {names[c1]}, {p2_type} for {names[c2]}."
    )


def world_misalignment(base: str, fx: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Real index of every currency against `base`, over the selected sample, at each country's latest month."""
    fx_s, p_s = fx.loc[start_ts:end_ts], prices.loc[start_ts:end_ts]
    if base not in fx_s or base not in p_s:
        return pd.DataFrame()
    others = fx_s.columns.intersection(p_s.columns).drop(base)
    s = fx_s[others].rdiv(fx_s[base], axis=0)  # base currency per 1 unit of each currency
    rer = s * p_s[others].div(p_s[base], axis=0)
    counts = rer.notna().sum()
    rer = rer.loc[:, counts >= 24]
    idx = 100 * rer / rer.mean()
    rows = []
    cutoff = end_ts - pd.DateOffset(months=24)
    for iso in idx.columns:
        col = idx[iso].dropna()
        if col.empty or col.index[-1] < cutoff:
            continue
        rows.append(dict(iso=iso, index=col.iloc[-1], month=col.index[-1], S=s.loc[col.index[-1], iso], via=""))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Euro members whose national series have ended inherit the Euro area's value.
    euro = out[out.iso == EURO_AREA]
    if not euro.empty and base not in EURO_MEMBERS | {EURO_AREA}:
        extra = [dict(euro.iloc[0], iso=m, via="via Euro area") for m in EURO_MEMBERS if m not in set(out.iso)]
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    if base == EURO_AREA:
        out = out[~out.iso.isin(EURO_MEMBERS)]
    out["mis"] = out["index"] - 100
    out["name"] = out.iso.map(lambda i: names.get(i, i))
    out["ccy"] = ["EUR" if via else ccy(i) for i, via in zip(out.iso, out.via)]
    return out


def section_map() -> None:
    st.markdown('<div class="section-title">How over- or undervalued is every currency right now?</div>',
                unsafe_allow_html=True)
    st.caption(f"Each country's real index against the chosen base currency, using the same sample period "
               f"({fmt_month(start_ts)}–{fmt_month(end_ts)}) to define RER*. Click a country to open that pair.")
    base = st.selectbox("Base currency", isos, index=isos.index(c2), format_func=label_for, key="map_base")
    try:
        with st.spinner("Loading every country from the IMF… (about 10 seconds the first time, then cached)"):
            fx, prices = fetch_world_panels()
    except requests.RequestException as exc:
        st.error(f"Could not download the world dataset from the IMF. Please try again.\n\n`{exc}`")
        return
    df = world_misalignment(base, fx, prices)
    if df.empty:
        st.warning("Not enough overlapping data for this base currency and sample period.")
        return

    e1, e2, e3 = st.columns(3)
    e1.metric("Currencies on the map", len(df))
    e2.metric("Median misalignment", f"{df.mis.median():+.1f}%")
    e3.metric(f"Share overvalued vs {ccy(base)}", f"{(df.mis > 0).mean() * 100:.0f}%")

    bound = 40
    hover = [
        f"<b>{r['name']}</b><br>Real index: {r['index']:.1f} "
        f"(<b>{r.mis:+.1f}%</b>)<br>1 {r.ccy} = {fmt_num(r.S)} {ccy(base)}<br>{fmt_month(r.month)}"
        + (f"<br><i>{r.via}</i>" if r.via else "")
        for _, r in df.iterrows()
    ]
    fig = go.Figure(go.Choropleth(
        locations=df.iso, z=df.mis.clip(-bound, bound), customdata=df.iso, text=hover,
        hovertemplate="%{text}<extra></extra>", zmid=0, zmin=-bound, zmax=bound,
        colorscale=MAP_SCALE, marker_line_color="#0B1020", marker_line_width=0.5,
        colorbar=dict(title=dict(text="vs avg"), ticksuffix="%", thickness=12, len=0.7,
                      tickvals=[-40, -20, 0, 20, 40], ticktext=["≤−40", "−20", "0", "+20", "≥+40"]),
    ))
    fig.update_geos(
        projection_type="natural earth", showframe=False, showcoastlines=False,
        showland=True, landcolor="#1A2140", showcountries=True, countrycolor="#0B1020",
        showocean=True, oceancolor="rgba(21,27,51,0.55)", bgcolor="rgba(0,0,0,0)", lataxis_range=[-58, 85],
    )
    base_layout(fig, 560, margin=dict(l=0, r=0, t=10, b=0), geo=dict(bgcolor="rgba(0,0,0,0)"))
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="world_map")

    points = (event or {}).get("selection", {}).get("points", []) if event else []
    if points:
        clicked = points[0].get("location") or points[0].get("customdata")
        if isinstance(clicked, list):
            clicked = clicked[0]
        if clicked in EURO_MEMBERS and clicked not in names:
            clicked = EURO_AREA
        if clicked in names and clicked != base:
            st.session_state.pending_pair = (clicked, base)
            st.rerun()

    n = min(10, len(df) // 2)
    # Both charts grow to the right from zero (bar length = size of the gap); the label carries the sign.
    over = df.nlargest(n, "mis").iloc[::-1]
    under = df.nsmallest(n, "mis").iloc[::-1]
    cl, cr = st.columns(2)
    for col, part, title, color in ((cl, over, "Most overvalued", COLOR_UP), (cr, under, "Most undervalued", COLOR_DOWN)):
        with col:
            st.markdown(f'<div class="section-title">{title} vs {ccy(base)}</div>', unsafe_allow_html=True)
            bar = go.Figure(go.Bar(
                x=part.mis.abs(), y=[nm[:28] for nm in part["name"]], orientation="h",
                marker=dict(color=color, cornerradius=6), text=[f"{m:+.0f}%" for m in part.mis],
                textposition="outside", customdata=part.mis, cliponaxis=False,
                hovertemplate="%{y}: %{customdata:+.1f}%<extra></extra>",
            ))
            base_layout(bar, 40 + 34 * n, margin=dict(l=10, r=50, t=4, b=4), showlegend=False)
            bar.update_xaxes(visible=False)
            st.plotly_chart(bar, width="stretch")

    st.caption("Colours are capped at ±40% so that hyperinflation episodes don't wash out the map; "
               "hover for exact values. Countries need ≥24 valid months in the sample and data within the last "
               "two years of it. Euro members show the Euro area's value.")


def section_verify() -> None:
    st.markdown('<div class="section-title">Step-by-step RER calculation</div>', unsafe_allow_html=True)
    st.caption("Pick any month to reproduce the numbers in the CSV by hand. Handy for the manual verification.")
    pick = st.select_slider("Month", options=list(v.index), value=v.index[-1], format_func=fmt_month)
    r = v.loc[pick]
    cA, cB = st.columns([2, 3])
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


def section_analytics() -> None:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Real volatility (ann.)", f"{vol_real:.1f}%", help="Std. dev. of monthly Δlog RER × √12.")
    a2.metric("Nominal volatility (ann.)", f"{vol_nom:.1f}%", help="Std. dev. of monthly Δlog S × √12.")
    a3.metric("Corr(ΔReal, ΔNominal)", f"{corr_rn:.2f}",
              help="Close to 1 under floating rates: nominal moves drive real moves because prices are sticky (Mussa, 1986).")
    a4.metric("AR(1) ρ of log RER", f"{rho:.3f}" if np.isfinite(rho) else "—")

    st.markdown('<div class="section-title">Where did the real change come from?</div>', unsafe_allow_html=True)
    st.caption(f"Δln RER = Δln S + Δln P1 − Δln P2, from {fmt_month(v.index[0])} to {fmt_month(v.index[-1])} "
               "(log points × 100 ≈ %).")
    first = v.iloc[0]
    d_s = 100 * math.log(last["S"] / first["S"])
    d_p1 = 100 * math.log(last["P1"] / first["P1"])
    d_p2 = -100 * math.log(last["P2"] / first["P2"])
    wf = go.Figure(go.Waterfall(
        x=[f"Nominal S ({ccy2}/{ccy1})", f"{names[c1]} prices (+P1)", f"{names[c2]} prices (−P2)", "Real exchange rate"],
        measure=["relative", "relative", "relative", "total"],
        y=[d_s, d_p1, d_p2, 0],
        text=[f"{d_s:+.1f}", f"{d_p1:+.1f}", f"{d_p2:+.1f}", f"{d_s + d_p1 + d_p2:+.1f}"],
        textposition="outside", connector=dict(line=dict(color="rgba(128,128,128,0.4)")),
        increasing=dict(marker=dict(color=COLOR_UP)), decreasing=dict(marker=dict(color=COLOR_DOWN)),
        totals=dict(marker=dict(color=COLOR_REAL)),
    ))
    st.plotly_chart(base_layout(wf, 380, showlegend=False, yaxis_title="log points × 100"), width="stretch")

    cL, cR = st.columns(2)
    with cL:
        st.markdown('<div class="section-title">Cumulative decomposition</div>', unsafe_allow_html=True)
        dec = pd.DataFrame({
            "Nominal S": 100 * np.log(v["S"] / first["S"]),
            "Inflation differential (P1/P2)": 100 * np.log((v["P1"] / v["P2"]) / (first["P1"] / first["P2"])),
            "Real exchange rate": 100 * np.log(v["RER"] / first["RER"]),
        })
        fd = go.Figure()
        for col, color in zip(dec.columns, [COLOR_NOMINAL, COLOR_PPP, COLOR_REAL]):
            fd.add_trace(go.Scatter(x=dec.index, y=dec[col], name=col, line=dict(color=color, width=2)))
        base_layout(fd, 360, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                    yaxis_title="log points × 100 since start")
        st.plotly_chart(fd, width="stretch")
    with cR:
        st.markdown('<div class="section-title">Distribution of the real index</div>', unsafe_allow_html=True)
        fh = go.Figure(go.Histogram(x=v["Real_Index"], nbinsx=40, marker=dict(color=COLOR_REAL, cornerradius=3),
                                    opacity=0.85))
        fh.add_vline(x=100, line=dict(color="gray", dash="dot"))
        fh.add_vline(x=float(last["Real_Index"]), line=dict(color=COLOR_PPP, width=2),
                     annotation_text="latest", annotation_position="top")
        st.plotly_chart(base_layout(fh, 360, bargap=0.05, xaxis_title="Real index", yaxis_title="Months"),
                        width="stretch")
        pct = float((v["Real_Index"] < last["Real_Index"]).mean() * 100)
        st.caption(f"The latest reading is higher than {pct:.0f}% of months in the sample.")

    hi, lo = v["Real_Index"].idxmax(), v["Real_Index"].idxmin()
    e1, e2, e3 = st.columns(3)
    e1.metric(f"Strongest real {ccy1}", f"{v.loc[hi, 'Real_Index']:.1f}", fmt_month(hi), delta_color="off")
    e2.metric(f"Weakest real {ccy1}", f"{v.loc[lo, 'Real_Index']:.1f}", fmt_month(lo), delta_color="off")
    e3.metric("Range of nominal S", f"{v['S'].min():,.4g} – {v['S'].max():,.4g}", f"{ccy2} per {ccy1}",
              delta_color="off")


def section_data() -> None:
    export = sample.reset_index(names="Date")
    export["Date"] = export["Date"].dt.strftime("%Y-%m")
    export.insert(1, "Country1", f"{names[c1]} ({c1})")
    export.insert(2, "Country2", f"{names[c2]} ({c2})")
    export = export[[
        "Date", "Country1", "Country2", "ER1_LCU_per_USD", "ER2_LCU_per_USD", "S", "P1", "P2",
        "RER", "RER_star", "Real_Index", "PPP_FairValue_S", "Misalignment_pct", "Valid_month",
    ]]
    d1, d2, d3 = st.columns(3)
    d1.metric("Months in sample", len(sample))
    d2.metric("Valid months", n_valid)
    d3.metric("Months with gaps", len(sample) - n_valid,
              help="A month is valid only if S, P1 and P2 are all available.")
    st.download_button(
        "⬇️  Download CSV (all observations and calculations)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"rer_{c1}_{c2}_{start_m}_{end_m}.csv",
        mime="text/csv", type="primary", width="stretch",
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


def section_method() -> None:
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

**World map.** The same calculation, repeated for every IMF country against the chosen base currency over the
same sample period, showing each country's latest real index minus 100.

**Data sources** (IMF SDMX 3.0, monthly, from {START_PERIOD}):
- Exchange rate: `{data_url("ER", FX_KEY.format(iso=c1))}`
- Exchange rate: `{data_url("ER", FX_KEY.format(iso=c2))}`
- Prices: `{data_url("CPI", (CPI_KEY if p1_type == "CPI" else HICP_KEY).format(iso=c1))}`
- Prices: `{data_url("CPI", (CPI_KEY if p2_type == "CPI" else HICP_KEY).format(iso=c2))}`

**Caveats.** IMF series can have gaps or end early (e.g. pre-euro national currencies stop in 1998, and the
Euro area HICP starts in December 1999). The Euro area publishes a harmonised index (HICP), which this app uses
automatically when no `CPI` series exists. Data is cached for six hours.
        """
    )


{
    SEC_MAP: section_map,
    SEC_CHART: section_chart,
    SECTIONS[2]: section_verify,
    SECTIONS[3]: section_analytics,
    SECTIONS[4]: section_data,
    SECTIONS[5]: section_method,
}[section]()

st.markdown(
    f"<div class='small-note' style='text-align:center;margin-top:2.5rem'>"
    f"Built with Streamlit · Data: International Monetary Fund · "
    f"Loaded {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')} · "
    f"The URL updates with your selections, so you can copy it to share this exact view.</div>",
    unsafe_allow_html=True,
)
