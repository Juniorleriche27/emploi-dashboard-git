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


# ---------- Assistant IA (Cohere) : NL -> plan JSON -> Pandas ----------
import os, json
try:
    import cohere
except Exception:
    cohere = None

def _apply_plan(df, plan):
    import numpy as np
    out = df.copy()

    # select
    if isinstance(plan.get("select"), list):
        ok = [c for c in plan["select"] if c in out.columns]
        if ok:
            out = out[ok]

    # where (liste de conditions simples)
    conds = plan.get("where", [])
    if isinstance(conds, list) and len(conds) > 0:
        mask = np.ones(len(out), dtype=bool)
        for cond in conds:
            col = cond.get("col"); op = cond.get("op"); val = cond.get("value")
            join = (cond.get("join") or "and").lower()
            if col not in out.columns or op not in ["==","!=",">","<",">=","<="]:
                continue
            s = out[col]
            try:
                if op == "==": cur = (s == val)
                elif op == "!=": cur = (s != val)
                elif op ==  ">": cur = (s  > val)
                elif op ==  "<": cur = (s  < val)
                elif op == ">=": cur = (s >= val)
                else:            cur = (s <= val)
            except Exception:
                continue
            mask = (mask & cur) if join == "and" else (mask | cur)
        out = out[mask]

    # groupby / agg
    if plan.get("groupby") and plan.get("agg"):
        gb = [c for c in plan["groupby"] if c in out.columns]
        agg_ok = {"count","sum","mean","median","min","max","std","nunique"}
        agg = {k:v for k,v in plan["agg"].items() if k in out.columns and v in agg_ok}
        if gb and agg:
            out = out.groupby(gb, dropna=False).agg(agg).reset_index()

    # sort
    if plan.get("sort_by"):
        by = [c for c in plan["sort_by"] if c in out.columns]
        if by:
            out = out.sort_values(by=by, ascending=bool(plan.get("ascending", False)))

    # top_n
    if isinstance(plan.get("top_n"), int):
        n = max(1, min(1000, int(plan["top_n"])))
        out = out.head(n)

    return out

with st.tabs(["Assistant IA (Cohere)"])[0]:
    st.subheader("Assistant IA (Cohere) — requête en langage naturel")
    key = (st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY"))
    if cohere is None:
        st.error("Le paquet `cohere` n’est pas installé. Ajoute `cohere>=5.3.0` dans requirements.txt.")
        st.stop()
    if not key:
        st.warning("Ajoute ta clé dans Settings → Secrets : COHERE_API_KEY.")
        st.stop()
    if 'df' not in globals() and 'df' not in locals():
        st.info("Charge d’abord un CSV dans l’onglet EDA, puis reviens ici.")
        st.stop()

    user_q = st.text_area(
        "Que veux-tu voir/obtenir ?",
        placeholder="Ex.: « moyenne du prix par catégorie pour 2023 », "
                    "« top 10 des régions par ventes », « évolution mensuelle du CA »..."
    )
    model = st.selectbox("Modèle Cohere", ["command-r", "command-r-plus"], index=0)

    schema = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
    sample = df.head(10).to_dict(orient="records")

    if st.button("Générer le résultat"):
        client = cohere.Client(key)

        system = (
            "Tu traduis la demande utilisateur en un **plan JSON** pour manipuler un DataFrame pandas nommé df. "
            "Réponds **UNIQUEMENT** avec un JSON valide suivant ce schéma strict :\n"
            "{\n"
            '  "select": [string]? ,\n'
            '  "where": [{"col": str, "op": "==|!=|>|<|>=|<=", "value": any, "join": "and|or"}]?,\n'
            '  "groupby": [string]? ,\n'
            '  "agg": {string: "count|sum|mean|median|min|max|std|nunique"}? ,\n'
            '  "sort_by": [string]? ,\n'
            '  "ascending": bool?,\n'
            '  "top_n": int? ,\n'
            '  "chart": {"type":"bar|line|hist","x":str,"y":str}?\n'
            "}\n"
            "Règles : n’invente pas de colonnes ; reste dans ce schéma ; pas d’explications en texte."
        )

        user = (
            f"Colonnes & dtypes: {schema}\n"
            f"Exemples (10 lignes): {sample}\n"
            f"Demande: {user_q}\n"
            "Réponds uniquement le JSON."
        )

        resp = client.chat(model=model, messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])

        # Compat v5 : .text ou .message.content[0].text
        if hasattr(resp, "text") and resp.text:
            raw = resp.text
        else:
            raw = resp.message.content[0].text

        try:
            plan = json.loads(raw)
        except Exception:
            st.error("Réponse non-JSON du modèle. Voici le retour brut :")
            st.code(raw)
            st.stop()

        st.write("**Plan généré**")
        st.code(plan, language="json")

        try:
            out = _apply_plan(df, plan)
        except Exception as e:
            st.error(f"Erreur lors de l’application du plan: {e}")
            st.stop()

        st.subheader("Résultat")
        st.dataframe(out, use_container_width=True)

        ch = plan.get("chart") or {}
        if {"type","x","y"} <= set(ch) and ch["x"] in out.columns and ch["y"] in out.columns:
            if ch["type"] == "line":
                st.line_chart(out.set_index(ch["x"])[ch["y"]])
            elif ch["type"] == "hist":
                st.bar_chart(out[ch["y"]].value_counts())
            else:  # bar
                st.bar_chart(out.set_index(ch["x"])[ch["y"]])
