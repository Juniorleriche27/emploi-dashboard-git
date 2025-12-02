import streamlit as st
import pandas as pd
import numpy as np

# --- dépendance en ligne : wbgapi (World Bank) ---
# Ajoute "wbgapi" dans requirements.txt (voir étape sous le code)

st.set_page_config(page_title="Emploi des jeunes — Analyse officielle (WDI)", layout="wide")
st.title("Emploi des jeunes — Analyse officielle (World Bank / ILO modelled)")
st.caption(
    "Source: World Development Indicators (WDI), indicateur SL.UEM.1524.ZS (taux de chômage des 15–24 ans). "
    "Les estimations sont modélisées (ILO). Utilisation pédagogique."
)

# -------- Paramètres utilisateur --------
COUNTRIES = {
    "Togo (TGO)": "TGO",
    "Côte d’Ivoire (CIV)": "CIV",
    "Bénin (BEN)": "BEN",
    "Ghana (GHA)": "GHA",
    "Sénégal (SEN)": "SEN",
    "France (FRA)": "FRA",
    "Maroc (MAR)": "MAR",
}
colA, colB = st.columns([1,1])
iso3 = colA.selectbox("Pays", list(COUNTRIES.keys()), index=0)
iso3 = COUNTRIES[iso3]
start, end = colB.slider("Période", 1991, 2024, (2000, 2024))

st.divider()

# -------- Chargement (API WDI) --------
@st.cache_data(ttl=86400)
def fetch_wdi_series(indicator: str, iso3: str, start: int, end: int) -> pd.DataFrame:
    import wbgapi as wb
    # labels=False => colonnes simples (codes) plutôt que libellés longs
    df = wb.data.DataFrame(indicator, economy=iso3, time=range(start, end + 1), labels=False)
    df = df.reset_index()

    # détecte de façon robuste les colonnes "économie / année / valeur"
    cols_str = [str(c) for c in df.columns]
    econ_col = "economy" if "economy" in cols_str else ("Economy" if "Economy" in cols_str else df.columns[0])
    time_col = "time"    if "time"    in cols_str else ("Time"    if "Time"    in cols_str else df.columns[1])

    value_cols = [c for c in df.columns if c not in [econ_col, time_col]]
    if not value_cols:  # si l'API renvoie vide pour la période
        return pd.DataFrame(columns=["year", "value"])

    val_col = value_cols[0]
    df = df.rename(columns={econ_col: "iso3", time_col: "year", val_col: "value"})
    df["year"] = df["year"].astype(int)
    df = df.loc[:, ["year", "value"]].dropna().sort_values("year")
    return df


INDICATOR = "SL.UEM.1524.ZS"  # youth unemployment (% ages 15–24)
data = fetch_wdi_series(INDICATOR, iso3, start, end)

if data.empty or len(data) < 2:
    st.warning("Pas assez de points sur cette période. Élargis la période.")
    st.stop()

# -------- Indicateurs analytiques --------
last = data.iloc[-1]
prev = data.iloc[-2]
delta_yoy = last["value"] - prev["value"]          # variation annuelle (points)
cagr_years = max(1, last["year"] - data.iloc[0]["year"])
cagr = ( (last["value"] + 1e-9) / (data.iloc[0]["value"] + 1e-9) ) ** (1 / cagr_years) - 1  # approx si valeurs >0
# pente (tendance linéaire)
x = data["year"].values
y = data["value"].values
slope = np.polyfit(x, y, 1)[0]  # points de % par an

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dernière valeur", f"{last['value']:.1f} %", f"{delta_yoy:+.1f} pt vs {int(prev['year'])}")
c2.metric("Tendance (pente/an)", f"{slope:+.2f} pt/an")
c3.metric("CAGR sur période", f"{100*cagr:+.2f} %/an")
c4.metric("Période couverte", f"{int(data['year'].min())}-{int(data['year'].max())}")

# lissage 3 ans + variation YoY
data["roll3"] = data["value"].rolling(3, min_periods=1).mean()
data["yoy"] = data["value"].diff(1)

st.subheader("Niveau & tendance (avec moyenne mobile 3 ans)")
st.line_chart(data.set_index("year")[["value", "roll3"]])

st.subheader("Variation annuelle (YoY, en points de pourcentage)")
st.bar_chart(data.set_index("year")[["yoy"]])

with st.expander("Données (téléchargement)"):
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Télécharger CSV (nettoyé)",
        data.to_csv(index=False).encode("utf-8"),
        file_name=f"youth_unemployment_{iso3}_{start}_{end}.csv",
        mime="text/csv",
    )

st.info(
    "Notes méthode — Source: WDI (World Bank), indicateur SL.UEM.1524.ZS.\n"
    "Niveau de preuve: **élevé** (source officielle), mais **incertitudes**: "
    "estimations modélisées par l’OIT, ruptures possibles dans les séries, comparabilité internationale imparfaite."
)
