import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Emploi-Jeunesse", layout="wide")

st.title("Dashboard Emploi-Jeunesse (prototype)")
st.caption("Données factices pour la démo – on branchera une source publique ensuite.")

# Mini dataset (démo)
data = pd.DataFrame({
    "mois": ["2024-06","2024-07","2024-08","2024-09","2024-10","2024-11"],
    "chomage": [20.1, 19.8, 19.6, 19.4, 19.3, 19.2],
    "activite": [40.2, 40.5, 40.9, 41.2, 41.4, 41.7],
    "emploi": [32.1, 32.5, 32.7, 33.2, 33.4, 33.6],
})

last, prev = data.iloc[-1], data.iloc[-2]
c1, c2, c3 = st.columns(3)
c1.metric("Taux de chômage jeunes", f"{last['chomage']:.1f} %", f"{last['chomage']-prev['chomage']:+.1f} pt")
c2.metric("Taux d’activité jeunes", f"{last['activite']:.1f} %", f"{last['activite']-prev['activite']:+.1f} pt")
c3.metric("Taux d’emploi jeunes",  f"{last['emploi']:.1f} %",  f"{last['emploi']-prev['emploi']:+.1f} pt")

st.subheader("Tendances récentes")
st.line_chart(data.set_index("mois")[["chomage","emploi","activite"]])

with st.expander("Voir les données"):
    st.dataframe(data, use_container_width=True)
