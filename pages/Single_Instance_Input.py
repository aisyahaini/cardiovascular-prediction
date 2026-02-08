import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import json

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
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

session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_model()

# =====================================================
# HELPER
# =====================================================
def predict_proba(x):
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: x.astype(np.float32)}
    )
    proba = outputs[1]
    if proba.ndim == 1:
        return proba
    return proba[:, 1]

# =====================================================
# UI
# =====================================================
st.set_page_config(layout="wide")
st.title("🫀 CVD Prediction – Single Input + Explainable AI")

st.subheader("🧾 Input Data Pasien")

input_data = {}

for feat in FEATURE_NAMES:
    input_data[feat] = st.number_input(
        label=feat,
        value=0.0
    )

x_input = np.array([[input_data[f] for f in FEATURE_NAMES]])

# =====================================================
# PREDICTION
# =====================================================
prob = predict_proba(x_input)[0]
pred = int(prob >= 0.5)

st.subheader("📌 Hasil Prediksi")
st.metric("Probabilitas CVD", f"{prob:.4f}")
st.metric("Prediksi", "CVD" if pred else "No CVD")

# =====================================================
# LIME-LIKE LOCAL EXPLANATION (MANUAL)
# =====================================================
st.subheader("🧩 LIME-like Local Explanation (Perturbation-based)")

delta = 0.1
lime_contrib = {}

base_prob = prob

for i, feat in enumerate(FEATURE_NAMES):
    x_perturb = x_input.copy()
    x_perturb[0, i] += delta
    new_prob = predict_proba(x_perturb)[0]
    lime_contrib[feat] = abs(new_prob - base_prob)

lime_df = (
    pd.DataFrame({
        "Feature": lime_contrib.keys(),
        "LIME_Local": lime_contrib.values()
    })
    .sort_values("LIME_Local", ascending=False)
)

st.dataframe(lime_df)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.barh(lime_df["Feature"], lime_df["LIME_Local"])
ax1.set_title("LIME-like Local Feature Importance")
ax1.invert_yaxis()
ax1.set_facecolor("white")
st.pyplot(fig1)

# =====================================================
# SHAP-LIKE GLOBAL EXPLANATION (MANUAL)
# =====================================================
st.subheader("📊 SHAP-like Global Explanation (Sampling-based)")

n_samples = 100
shap_contrib = {f: [] for f in FEATURE_NAMES}

for _ in range(n_samples):
    noise = np.random.normal(0, 0.1, size=x_input.shape)
    x_sample = x_input + noise
    base = predict_proba(x_sample)[0]

    for i, feat in enumerate(FEATURE_NAMES):
        x_p = x_sample.copy()
        x_p[0, i] += delta
        new_p = predict_proba(x_p)[0]
        shap_contrib[feat].append(abs(new_p - base))

shap_df = (
    pd.DataFrame({
        "Feature": shap_contrib.keys(),
        "SHAP_Global": [np.mean(v) for v in shap_contrib.values()]
    })
    .sort_values("SHAP_Global", ascending=False)
)

st.dataframe(shap_df)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.barh(shap_df["Feature"], shap_df["SHAP_Global"])
ax2.set_title("SHAP-like Global Feature Importance")
ax2.invert_yaxis()
ax2.set_facecolor("white")
st.pyplot(fig2)

# =====================================================
# CONSISTENCY ANALYSIS
# =====================================================
st.subheader("📐 Konsistensi SHAP-like vs LIME-like")

comparison = shap_df.merge(lime_df, on="Feature")

comparison["SHAP_Rank"] = comparison["SHAP_Global"].rank(ascending=False)
comparison["LIME_Rank"] = comparison["LIME_Local"].rank(ascending=False)

rho, _ = spearmanr(comparison["SHAP_Rank"], comparison["LIME_Rank"])
tau, _ = kendalltau(comparison["SHAP_Rank"], comparison["LIME_Rank"])

col1, col2 = st.columns(2)
col1.metric("Spearman ρ", f"{rho:.3f}")
col2.metric("Kendall τ", f"{tau:.3f}")

st.dataframe(comparison.sort_values("SHAP_Rank"))

# =====================================================
# INTERPRETASI
# =====================================================
st.subheader("📌 Interpretasi")

st.markdown("""
- **LIME-like** menunjukkan fitur yang paling memengaruhi **satu pasien**.
- **SHAP-like** menunjukkan fitur yang stabil dan penting secara **global**.
- Korelasi tinggi → konsistensi interpretasi model baik.
- Pendekatan ini **model-agnostic dan bebas dependensi SHAP/LIME library**.
""")
