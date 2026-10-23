# 💱 Real Exchange Rate Explorer

An interactive Streamlit app (FINC 470) that computes the **bilateral real exchange rate** between any two
IMF countries, or the Euro area, using monthly IMF CPI and exchange-rate data from January 1976.

**Live app:** _add your streamlit.app link here_

## Economics

| Symbol | Definition |
|---|---|
| `ERi(t)` | IMF exchange rate: country *i* local currency per USD (`XDC_USD`) |
| `S(t)` | Country 2 currency per 1 unit of Country 1 currency = `ER2(t) / ER1(t)` |
| `P1(t)`, `P2(t)` | Monthly CPI of Country 1 and Country 2 (HICP for the Euro area) |
| `RER(t)` | `S(t) × P1(t) / P2(t)` |
| `RER*` | Average of `RER(t)` over valid months in the selected sample |
| `Real Index(t)` | `100 × RER(t) / RER*`. Averages exactly 100 over the sample |
| `PPP fair value` | `RER* × P2(t) / P1(t)`, the nominal rate that would put the real index at 100 |

Example: Country 1 = United Kingdom, Country 2 = United States → `S` is **USD per GBP**.

## Features

- **World map** (home page): every currency's real over/undervaluation against a chosen base, with
  most over/undervalued rankings. Click a country to open that pair
- Any pair of IMF countries plus the Euro area (`G163`), with a swap button and quick-pick presets
- Sample-period slider with Full / Euro-era / 20y / 10y / 5y shortcuts. `RER*` is recomputed on every change
- **Real**, **Nominal** or **Both** views (Both uses separate y-axes). Nominal `S(t)` is plotted in actual currency units
- Over/undervaluation shading and a PPP fair-value line on the nominal chart
- **Verify a month** tab: a step-by-step worked RER calculation for any month, plus a live check that the index averages 100
- **Analytics** tab: half-life of PPP deviations (AR(1)), real vs nominal volatility, the real/nominal correlation,
  a waterfall decomposition `Δln RER = Δln S + Δln P1 − Δln P2`, and the distribution of the index
- CSV download of every observation and calculation in the selected sample
- Shareable URLs: the address bar stores the current pair, dates and view

## Data

IMF SDMX 3.0 API (`https://api.imf.org/external/sdmx/3.0`), no API key required.

- CPI: dataflow `CPI`, key `{ISO3}.CPI._T.IX.M`, falling back to `{ISO3}.HICP._T.IX.M` (the Euro area only publishes HICP)
- Exchange rate: dataflow `ER`, key `{ISO3}.XDC_USD.PA_RT.M`

Coverage varies. For example, the Euro area HICP in the IMF database starts in December 1999, and pre-euro national
currencies (e.g. DEM, FRF) end in December 1998. Only months where S, P1 and P2 all exist count as valid.

## Files

- `app.py`: the Streamlit app
- `requirements.txt`: Python dependencies
- `README.md`: this file
- `.streamlit/config.toml`: optional dark colour theme (the app works without it)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Built for Python 3.12 and deployed on Streamlit Community Cloud.
