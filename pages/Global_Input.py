import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# =====================================================
# HELPER
# =====================================================
def extract_proba(outputs):
    proba_raw = outputs[1]

    if isinstance(proba_raw[0], dict):
        return np.array([[p[0], p[1]] for p in proba_raw])

    if np.ndim(proba_raw) == 1:
        return np.column_stack([1 - proba_raw, proba_raw])

    return np.array(proba_raw)


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("feature_names.json") as f:
        feature_names = json.load(f)

    return (
        session,
        feature_names,
        session.get_inputs()[0].name,
        [o.name for o in session.get_outputs()]
    )


session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_onnx_model()

st.title("📂 Prediksi Penyakit Cardiovascular (ONNX + XAI)")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# =====================================================
# MAIN
# =====================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # =====================
    # PREPROCESS
    # =====================
    mapping_binary = {
        "sex": {"male": 1, "female": 0},
        "exang": {"yes": 1, "no": 0},
        "fbs": {"true": 1, "false": 0}
    }

    for col, mp in mapping_binary.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(mp)

    X_raw = df.drop(columns=["num", "id"], errors="ignore")

    shap_to_df_mapping = {
        "num_major_vessels": "ca",
        "chest_pain_type": "cp",
        "thalassemia_type": "thal",
        "st_slope_type": "slope",
        "cholesterol": "chol",
        "st_depression": "oldpeak",
        "resting_blood_pressure": "trestbps",
        "sex": "sex",
        "age": "age",
        "max_heart_rate_achieved": "thalch",
        "Restecg": "restecg",
        "exercise_induced_angina": "exang",
        "fasting_blood_sugar": "fbs",
        "country": "dataset"
    }

    X_model = X_raw.rename(columns={v: k for k, v in shap_to_df_mapping.items()})

    for col in FEATURE_NAMES:
        if col not in X_model.columns:
            X_model[col] = 0

    X_model = X_model[FEATURE_NAMES].fillna(0).astype(np.float32)

    # =====================
    # PREDICTION
    # =====================
    outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: X_model.values})
    df["prediction"] = outputs[0]
    df["probability"] = extract_proba(outputs)[:, 1]

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================================================
    # LIME EXPLAINER
    # =====================================================
    def onnx_predict_proba(x):
        return extract_proba(
            session.run(OUTPUT_NAMES, {INPUT_NAME: x.astype(np.float32)})
        )

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_model.values,
        feature_names=FEATURE_NAMES,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    st.subheader("🧩 LIME – Local Explanation")
    idx = st.slider("Pilih indeks data", 0, len(X_model) - 1, 0)

    exp = lime_explainer.explain_instance(
        X_model.iloc[idx].values,
        onnx_predict_proba,
        num_features=len(FEATURE_NAMES)
    )

    # 👉 FIX BACKGROUND PUTIH
    html = exp.as_html().replace(
        "<body>",
        "<body style='background-color:white;'>"
    )
    st.components.v1.html(html, height=650, scrolling=True)

    lime_weights = dict(exp.as_list())

    # =====================================================
    # SHAP-LIKE GLOBAL
    # =====================================================
    st.subheader("📊 Global Feature Importance (SHAP-like)")

    feature_importance = defaultdict(list)

    for i in np.random.choice(len(X_model), min(50, len(X_model)), replace=False):
        e = lime_explainer.explain_instance(
            X_model.iloc[i].values,
            onnx_predict_proba,
            num_features=len(FEATURE_NAMES)
        )
        for f, w in e.as_list():
            feature_importance[f.split(" ")[0]].append(abs(w))

    shap_like = {f: np.mean(w) for f, w in feature_importance.items()}

    shap_df = (
        pd.DataFrame({
            "Feature": shap_like.keys(),
            "SHAP_like_Importance": shap_like.values()
        })
        .sort_values("SHAP_like_Importance", ascending=False)
    )

    st.dataframe(shap_df)

    # =====================
    # SHAP BAR PLOT (WITH VALUES)
    # =====================
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shap_df["Feature"], shap_df["SHAP_like_Importance"])

    for i, v in enumerate(shap_df["SHAP_like_Importance"]):
        ax.text(v, i, f"{v:.4f}", va="center")

    ax.set_xlabel("Mean |Contribution|")
    ax.set_title("Global Feature Importance (SHAP-like)")
    ax.invert_yaxis()
    ax.set_facecolor("white")

    st.pyplot(fig)

    # =====================================================
    # LIME vs SHAP COMPARISON
    # =====================================================
    st.subheader("📈 Perbandingan LIME vs SHAP-like")

    comparison_df = pd.DataFrame({
        "Feature": shap_df["Feature"],
        "LIME_Local_Weight": [
            abs(lime_weights.get(f, 0)) for f in shap_df["Feature"]
        ],
        "SHAP_like_Importance": shap_df["SHAP_like_Importance"]
    })

    st.dataframe(comparison_df)

    # =====================
    # COMPARISON PLOT
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    ax.bar(x - 0.2, comparison_df["LIME_Local_Weight"], width=0.4, label="LIME")
    ax.bar(x + 0.2, comparison_df["SHAP_like_Importance"], width=0.4, label="SHAP-like")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Feature"], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Perbandingan Feature Importance")
    ax.legend()
    ax.set_facecolor("white")

    st.pyplot(fig)

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
