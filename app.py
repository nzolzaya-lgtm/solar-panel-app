import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Solar Panel Calculator for Mongolia", page_icon="☀️", layout="wide")

TILTS = [0, 15, 30, 45, 60, 75, 90]
SNOW_MONTHS = [11, 12, 1, 2, 3]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    m = pd.read_csv("solar_monthly.csv")
    m = m.dropna(subset=["kwh_per_kwp"])
    full = m.groupby(["location", "year", "tilt"])["month"].transform("count") == 12
    annual = m[full].groupby(["location", "year", "tilt"])["kwh_per_kwp"].sum().reset_index()
    by_tilt = annual.groupby(["location", "tilt"])["kwh_per_kwp"].mean().reset_index()
    best = by_tilt.loc[by_tilt.groupby("location")["kwh_per_kwp"].idxmax()].rename(
        columns={"tilt": "best_tilt", "kwh_per_kwp": "yield_best"})
    locs = pd.read_csv("locations.csv")
    best = best.merge(locs, on="location").sort_values("yield_best", ascending=False).reset_index(drop=True)
    return m, annual, by_tilt, best


# ---------------------------------------------------------------
# 2. CALCULATIONS
# ---------------------------------------------------------------
def monthly_energy(m, loc, tilt, kw, pr, snow):
    d = m[(m["location"] == loc) & (m["tilt"] == tilt)]
    avg = d.groupby("month")["kwh_per_kwp"].mean()
    factor = pd.Series({mo: (1 - snow if mo in SNOW_MONTHS else 1.0) for mo in avg.index})
    return (avg * factor * kw * pr)


def cash_flows(e1, cost, tariff, growth, degr, maint, self_use, years):
    flows = [-cost]
    for t in range(1, years + 1):
        energy = e1 * (1 - degr) ** (t - 1)
        flows.append(energy * self_use * tariff * (1 + growth) ** (t - 1) - cost * maint)
    return np.array(flows)


def payback_years(flows):
    cum = np.cumsum(flows)
    if not (cum >= 0).any():
        return np.nan
    i = int(np.argmax(cum >= 0))
    return (i - 1) + (-cum[i - 1] / flows[i]) if i > 0 else 0.0


def npv(rate, flows):
    return float(sum(f / (1 + rate) ** t for t, f in enumerate(flows)))


def irr(flows):
    lo, hi = -0.9, 5.0
    if npv(lo, flows) * npv(hi, flows) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo, flows) * npv(mid, flows) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


if not (os.path.exists("solar_monthly.csv") and os.path.exists("locations.csv")):
    st.error("The data files `solar_monthly.csv` and `locations.csv` are missing. "
             "Run the notebook first and upload both files to the same GitHub folder as app.py.")
    st.stop()

m, annual, by_tilt, best = load_data()
LOCS = sorted(m["location"].unique())

# ---------------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("☀️ Your solar system")
loc = st.sidebar.selectbox("Location", LOCS, index=LOCS.index("Ulaanbaatar") if "Ulaanbaatar" in LOCS else 0)
kw = st.sidebar.slider("System size (kW)", 1.0, 50.0, 3.0, step=0.5)
use_best = st.sidebar.checkbox("Use the best tilt for this location", value=True)
best_tilt = int(best.loc[best["location"] == loc, "best_tilt"].iloc[0])
if use_best:
    tilt = best_tilt
    st.sidebar.caption(f"Best tilt here: {best_tilt} degrees")
else:
    tilt = st.sidebar.select_slider("Panel tilt (degrees)", options=TILTS, value=45)

st.sidebar.subheader("Money (replace the examples!)")
cost_per_kw = st.sidebar.number_input("System cost (MNT per kW)", 500_000, 20_000_000, 3_500_000, step=100_000,
                                      help="Example value. Replace it with a real quote or report.")
tariff = st.sidebar.number_input("Electricity tariff (MNT per kWh)", 50, 2000, 250, step=10,
                                 help="Example value. Replace it with the real tariff.")
growth = st.sidebar.slider("Tariff growth per year (%)", 0.0, 15.0, 5.0, step=0.5) / 100
self_use = st.sidebar.slider("Share of the energy you use yourself (%)", 20, 100, 100, step=5) / 100

with st.sidebar.expander("Technical assumptions"):
    pr = st.slider("Performance ratio", 0.60, 0.95, 0.80, step=0.01,
                   help="Losses from cables, dust and the inverter. Typical: 0.75 to 0.85.")
    snow = st.slider("Winter snow loss (Nov-Mar, %)", 0, 30, 5) / 100
    degr = st.slider("Panel degradation per year (%)", 0.0, 2.0, 0.5, step=0.1) / 100
    maint = st.slider("Maintenance per year (% of system cost)", 0.0, 3.0, 1.0, step=0.1) / 100
    disc = st.slider("Discount rate for NPV (%)", 0.0, 25.0, 10.0, step=0.5) / 100
    life = st.slider("System lifetime (years)", 10, 30, 25)

cost = kw * cost_per_kw
me = monthly_energy(m, loc, tilt, kw, pr, snow)
e1 = float(me.sum())
flows = cash_flows(e1, cost, tariff, growth, degr, maint, self_use, life)
pb = payback_years(flows)
npv_val = npv(disc, flows)
irr_val = irr(flows)
year1_saving = e1 * self_use * tariff

st.title("☀️ Solar Panel Calculator for Mongolia")
st.markdown("**Question:** How much electricity can a solar panel system make at my location in Mongolia, "
            "how much money does it save, and how many years until it pays for itself?")
st.warning("The cost and tariff values in the sidebar are **examples**. Replace them with real, sourced values "
           "before you trust the money results.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Result", "Map and ranking", "Best tilt", "Money and what-if", "About the data"])

# ---------------------------------------------------------------
# TAB 1: RESULT
# ---------------------------------------------------------------
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Energy in year 1", f"{e1:,.0f} kWh")
    c2.metric("Savings in year 1", f"{year1_saving / 1e6:,.2f} M MNT")
    c3.metric("Payback", "never" if np.isnan(pb) else f"{pb:.1f} years")
    c4.metric(f"NPV ({life} years)", f"{npv_val / 1e6:,.1f} M MNT")
    c5.metric("IRR", "n/a" if np.isnan(irr_val) else f"{irr_val * 100:.1f}% per year")
    st.caption(f"{kw:g} kW in {loc}, tilt {tilt} degrees, facing south. System cost: {cost / 1e6:,.1f} M MNT.")

    left, right = st.columns(2)
    dfm = pd.DataFrame({"Month": [MONTH_NAMES[i - 1] for i in me.index], "kWh": me.values})
    fig = px.bar(dfm, x="Month", y="kWh", title="Energy by month (average year)",
                 color="kWh", color_continuous_scale="YlOrRd")
    fig.update_layout(coloraxis_showscale=False)
    left.plotly_chart(fig, width="stretch")
    left.caption(f"Best month: {MONTH_NAMES[int(me.idxmax()) - 1]} ({me.max():,.0f} kWh). "
                 f"Worst month: {MONTH_NAMES[int(me.idxmin()) - 1]} ({me.min():,.0f} kWh). "
                 "Winter is lower because the days are short and the sun is low.")

    cum = np.cumsum(flows) / 1e6
    fig = go.Figure(go.Scatter(x=list(range(len(cum))), y=cum, mode="lines", line=dict(color="seagreen", width=3)))
    fig.add_hline(y=0, line_color="black")
    if not np.isnan(pb):
        fig.add_vline(x=pb, line_dash="dash", line_color="red", annotation_text=f"Payback {pb:.1f} years")
    fig.update_layout(title="Cumulative cash flow", xaxis_title="Year", yaxis_title="Million MNT")
    right.plotly_chart(fig, width="stretch")
    right.caption("The line crosses zero when the system has paid for itself. After that, the savings are profit.")

    yr = annual[(annual["location"] == loc) & (annual["tilt"] == tilt)].copy()
    if not yr.empty:
        yr["year"] = yr["year"].astype(str)
        fig = px.bar(yr, x="year", y="kwh_per_kwp", title="Does the energy change from year to year? (per kWp)",
                     labels={"kwh_per_kwp": "kWh per kWp", "year": "Year"})
        fig.add_hline(y=yr["kwh_per_kwp"].mean(), line_dash="dash", annotation_text="Average")
        st.plotly_chart(fig, width="stretch")
        spread = (yr["kwh_per_kwp"].max() - yr["kwh_per_kwp"].min()) / yr["kwh_per_kwp"].mean() * 100
        st.caption(f"The best and worst year differ by {spread:.1f}%. A small difference means one average year is a good estimate.")

# ---------------------------------------------------------------
# TAB 2: MAP AND RANKING
# ---------------------------------------------------------------
with tab2:
    fig = px.scatter_geo(best, lat="lat", lon="lon", color="yield_best", size="yield_best",
                         hover_name="location", hover_data={"aimag": True, "best_tilt": True,
                                                            "yield_best": ":.0f", "lat": False, "lon": False},
                         color_continuous_scale="YlOrRd", scope="asia",
                         title="Yearly energy per kWp at the best tilt (each dot is an aimag center)",
                         labels={"yield_best": "kWh per kWp per year"})
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="gray", showland=True,
                    landcolor="rgb(245,245,235)")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Darker dots get more sunlight energy per kW of panels. Differences between places are small compared "
               "with differences between seasons.")

    r = best.sort_values("yield_best")
    fig = px.bar(r, x="yield_best", y="location", orientation="h", title="Ranking of the locations",
                 labels={"yield_best": "kWh per kWp per year", "location": ""}, color="yield_best",
                 color_continuous_scale="YlOrRd")
    fig.update_layout(coloraxis_showscale=False, height=650)
    st.plotly_chart(fig, width="stretch")
    spread = (best["yield_best"].max() - best["yield_best"].min()) / best["yield_best"].mean() * 100
    st.caption(f"The best and the weakest location differ by {spread:.1f}%.")

# ---------------------------------------------------------------
# TAB 3: BEST TILT
# ---------------------------------------------------------------
with tab3:
    chosen = st.multiselect("Locations to compare", LOCS, default=[loc])
    if chosen:
        d = by_tilt[by_tilt["location"].isin(chosen)]
        fig = px.line(d, x="tilt", y="kwh_per_kwp", color="location", markers=True,
                      title="Yearly energy per kWp by panel tilt (facing south)",
                      labels={"tilt": "Tilt (0 = flat, 90 = vertical)", "kwh_per_kwp": "kWh per kWp per year"})
        st.plotly_chart(fig, width="stretch")
        s = by_tilt[by_tilt["location"] == loc].set_index("tilt")["kwh_per_kwp"]
        st.caption(f"In {loc}, the best tilt is {best_tilt} degrees. A flat panel makes "
                   f"{(1 - s.get(0, np.nan) / s.max()) * 100:.0f}% less and a vertical panel makes "
                   f"{(1 - s.get(90, np.nan) / s.max()) * 100:.0f}% less than the best tilt.")

    mm = m[(m["location"] == loc) & (m["tilt"].isin([0, 45, 90]))].groupby(["tilt", "month"])["kwh_per_kwp"].mean().reset_index()
    mm["tilt"] = mm["tilt"].astype(str) + " degrees"
    fig = px.line(mm, x="month", y="kwh_per_kwp", color="tilt", markers=True,
                  title=f"{loc}: energy by month for three tilts",
                  labels={"month": "Month", "kwh_per_kwp": "kWh per kWp"})
    st.plotly_chart(fig, width="stretch")
    st.caption("A steep panel is better in winter (low sun), a flat panel is better in summer (high sun).")

# ---------------------------------------------------------------
# TAB 4: MONEY
# ---------------------------------------------------------------
with tab4:
    st.subheader("Payback in every location (same system and settings)")
    rows = []
    for L in LOCS:
        e = float(monthly_energy(m, L, tilt, kw, pr, snow).sum())
        f = cash_flows(e, cost, tariff, growth, degr, maint, self_use, life)
        rows.append({"Location": L, "Energy (kWh)": round(e), "Payback (years)": round(payback_years(f), 1),
                     "NPV (M MNT)": round(npv(disc, f) / 1e6, 1)})
    loc_table = pd.DataFrame(rows).sort_values("Payback (years)").reset_index(drop=True)
    fig = px.bar(loc_table.sort_values("Payback (years)", ascending=False), x="Payback (years)", y="Location",
                 orientation="h", title=f"Payback time by location ({kw:g} kW, tilt {tilt})")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(loc_table, width="stretch", hide_index=True)

    st.subheader("What if the tariff or the cost is different?")
    mult = [0.6, 0.8, 1.0, 1.2, 1.5]
    grid = pd.DataFrame(index=[f"{int(c * 100)}%" for c in mult], columns=[f"{int(t * 100)}%" for t in mult], dtype=float)
    for c in mult:
        for t in mult:
            f = cash_flows(e1, cost * c, tariff * t, growth, degr, maint, self_use, life)
            grid.loc[f"{int(c * 100)}%", f"{int(t * 100)}%"] = payback_years(f)
    fig = px.imshow(grid.values.astype(float), x=list(grid.columns), y=list(grid.index), text_auto=".1f",
                    color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels={"x": "Tariff (compared with your setting)", "y": "System cost (compared with your setting)",
                            "color": "Years"}, title="Payback in years")
    st.plotly_chart(fig, width="stretch")
    st.caption("The tariff and the cost are the least certain numbers, so this table shows how much the payback changes. "
               "Cheaper systems and higher tariffs shorten the payback.")

# ---------------------------------------------------------------
# TAB 5: ABOUT
# ---------------------------------------------------------------
with tab5:
    yrs = f"{int(m['year'].min())} to {int(m['year'].max())}"
    st.markdown(f"""
### About the data
- **Source:** Open-Meteo Historical Weather API (ERA5 reanalysis, Copernicus / ECMWF): hourly global, direct and diffuse sunlight and air temperature. https://open-meteo.com/en/docs/historical-weather-api
- **Locations:** {len(LOCS)} places (Ulaanbaatar and the 21 aimag centers), approximate coordinates.
- **Years:** {yrs}.
- **License:** free for non-commercial use under CC BY 4.0. Credit: Open-Meteo.

### How the energy is calculated (done in the notebook)
1. Cleaned the hourly data: parsed dates, removed duplicate hours and missing values, set negative sunlight to 0, filled short temperature gaps.
2. Calculated the sun position for every hour with `pvlib`.
3. Calculated the sunlight on a tilted panel facing south (direct + diffuse + ground reflection, isotropic sky model, ground reflection 0.20) for tilts 0, 15, 30, 45, 60, 75 and 90 degrees.
4. Adjusted for panel temperature (about -0.4% power per degree C above 25 C, panel is warmer than the air in sunlight).
5. Added up the energy per kWp for each month.
6. In this app: energy = kWh per kWp × system size × performance ratio × (1 - snow loss in Nov-Mar).

### How the money is calculated
- Savings = energy used × tariff. The tariff grows each year. The panels lose power each year (degradation). Maintenance is a share of the system cost each year.
- **Payback:** the year the total cash flow becomes positive. **NPV:** value today of all cash flows. **IRR:** yearly return of the investment.

### Limits
- ERA5 sunlight is a model estimate on a grid of about 30 km. It can differ from a real measurement on your roof.
- Ground reflection is fixed at 0.20, but snow reflects more, so real winter energy may be a bit higher.
- Snow on the panels is only a simple percentage loss. Shade, dust and the real inverter are not modeled.
- Cost and tariff values in the sidebar are examples. Replace them with cited values.
- The model assumes you use the energy yourself. Rules for selling extra electricity to the grid are not included.
- Panels face south only.
""")
    st.dataframe(m.head(50), width="stretch")
