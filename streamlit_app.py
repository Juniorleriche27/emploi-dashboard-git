import io
import numpy as np
import pandas as pd
import streamlit as st

# (optionnel) plots avancés
try:
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

st.set_page_config(page_title="Kaggle EDA Pro — Analyse exploratoire", layout="wide")
st.title("Kaggle EDA Pro — Analyse exploratoire robuste")
st.caption(
    "Charge un dataset Kaggle (CSV) puis explore : résumé, valeurs manquantes, distributions, "
    "corrélations, séries temporelles, catégories. Export du CSV nettoyé."
)

# ------------- Sidebar: chargement & options -------------
with st.sidebar:
    st.header("Chargement")
    f = st.file_uploader("CSV Kaggle", type=["csv"])
    sep = st.selectbox("Séparateur", [",", ";", "|", r"\t"], index=0)
    dec = st.selectbox("Décimal", [".", ","], index=0)
    enc = st.selectbox("Encodage", ["utf-8", "latin1", "utf-16"], index=0)
    st.caption("Astuce: si erreur de décodage, change l’encodage et relance le chargement.")

def load_csv(file, sep, dec, enc):
    if file is None:
        return None, "Aucun fichier"
    # Première tentative
    try:
        df = pd.read_csv(file, sep=sep.encode('utf-8').decode('unicode_escape'), decimal=dec, encoding=enc)
        return df, "OK"
    except UnicodeDecodeError:
        # Tentative fallback
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=sep.encode('utf-8').decode('unicode_escape'), decimal=dec, encoding="latin1")
            return df, "OK (fallback latin1)"
        except Exception as e:
            return None, f"Erreur décodage: {e}"
    except Exception as e:
        return None, f"Erreur lecture: {e}"

df, status = load_csv(f, sep, dec, enc)

# ------------- Fonctions utilitaires -------------
def guess_datetime_columns(df: pd.DataFrame, thresh: float = 0.8):
    """Renvoie la liste des colonnes qui semblent être des dates (≥ thresh parsables)."""
    dt_cols = []
    for c in df.columns:
        s = df[c]
        ok = 0
        n = min(len(s), 300)  # échantillon
        if n == 0: 
            continue
        sample = s.dropna().astype(str).head(n)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            ok = parsed.notna().mean()
        except Exception:
            ok = 0
        if ok >= thresh:
            dt_cols.append(c)
    return dt_cols

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normaliser noms colonnes
    out.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in out.columns]
    return out

def bytes_download(data: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    data.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------- Corps principal -------------
if df is None:
    st.info("➡️ Charge un CSV Kaggle dans la barre latérale pour commencer.")
    st.stop()

df = basic_clean(df)
st.success(f"Fichier chargé ({status}) — shape: {df.shape[0]} lignes × {df.shape[1]} colonnes")

tabs = st.tabs(["Aperçu", "Qualité & manquants", "Numérique", "Corrélations", "Séries temporelles", "Catégories", "Exporter"])

with tabs[0]:
    st.subheader("Aperçu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", f"{len(df):,}")
    c2.metric("Colonnes", f"{df.shape[1]}")
    c3.metric("Valeurs manquantes", f"{int(df.isna().sum().sum()):,}")
    st.dataframe(df.head(50), use_container_width=True)

with tabs[1]:
    st.subheader("Qualité & manquants")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_ratio = (miss / len(df) * 100).round(1)
    qual = pd.DataFrame({"missing": miss, "missing_%": miss_ratio})
    st.dataframe(qual, use_container_width=True)
    if PLOTLY and len(qual) > 0:
        fig = px.bar(qual.reset_index().rename(columns={"index":"col"}), x="col", y="missing_%", title="Taux de manquants (%)")
        fig.update_layout(xaxis={"tickangle": -45})
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Exploration numérique")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("Aucune colonne numérique détectée.")
    else:
        target = st.selectbox("Colonne numérique à explorer", num_cols)
        desc = df[num_cols].describe().T
        st.dataframe(desc, use_container_width=True)
        if PLOTLY:
            fig = px.histogram(df, x=target, nbins=50, title=f"Distribution de {target}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df[target])

with tabs[3]:
    st.subheader("Corrélations (numériques)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Besoin d’au moins 2 colonnes numériques.")
    else:
        corr = df[num_cols].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)
        if PLOTLY:
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matrice de corrélation (Pearson)")
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Séries temporelles")
    dt_cols = guess_datetime_columns(df)
    if not dt_cols:
        st.info("Aucune colonne date détectée automatiquement. Sélectionne manuellement si besoin.")
        manual_dt = st.selectbox("Choisir une colonne date (optionnel)", [None] + df.columns.tolist())
        if manual_dt:
            dt_cols = [manual_dt]
    if dt_cols:
        dt_col = st.selectbox("Colonne date", dt_cols)
        metric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not metric_cols:
            st.warning("Aucune colonne numérique à tracer.")
        else:
            ycol = st.selectbox("Série à tracer", metric_cols)
            tmp = df[[dt_col, ycol]].dropna().copy()
            tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
            tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)
            # agrégation mensuelle si haute fréquence
            if tmp[dt_col].diff().dt.days.median() is not None and tmp[dt_col].diff().dt.days.median() < 25:
                tmp = tmp.set_index(dt_col).resample("MS")[ycol].mean().reset_index()
            if PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tmp[dt_col], y=tmp[ycol], mode="lines", name=ycol))
                rm = tmp[ycol].rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=tmp[dt_col], y=rm, mode="lines", name=f"{ycol} (MM3)", line=dict(dash="dash")))
                fig.update_layout(title=f"Tendance {ycol}", xaxis_title="Date", yaxis_title=ycol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(tmp.set_index(dt_col)[ycol])
            # variations
            tmp["MoM"] = tmp[ycol].diff()
            tmp["YoY"] = tmp[ycol].diff(12)
            st.write("Variations (Δ MoM / Δ YoY) — premières lignes :")
            st.dataframe(tmp.tail(24), use_container_width=True)
    else:
        st.warning("Aucune colonne date sélectionnée/détectée.")

with tabs[5]:
    st.subheader("Analyse catégorielle")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.warning("Aucune colonne catégorielle détectée.")
    else:
        cat = st.selectbox("Catégorie", cat_cols)
        topn = st.slider("Top N", 5, 50, 15)
        vc = df[cat].astype(str).value_counts().head(topn)
        st.dataframe(vc.to_frame("compte"), use_container_width=True)
        if PLOTLY:
            st.plotly_chart(px.bar(vc[::-1], title=f"Top {topn} {cat}").update_layout(yaxis_title="compte"),
                            use_container_width=True)

with tabs[6]:
    st.subheader("Exporter")
    st.download_button("Télécharger CSV nettoyé", data=bytes_download(df), file_name="dataset_nettoye.csv", mime="text/csv")
    st.caption("Le CSV nettoyé garde uniquement les colonnes et le formatage normalisés.")
