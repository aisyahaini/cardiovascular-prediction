import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import spearmanr, kendalltau

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
# DEFAULT VALUES (AMAN & REALISTIS)
# =====================================================
DEFAULT_VALUES = {
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# =====================================================
# SAFE PROBABILITY EXTRACTOR
# =====================================================
def predict_proba(x):
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: x.astype(np.float32)}
    )

    proba_raw = outputs[1]

    # sklearn → ONNX (list of dict)
    if isinstance(proba_raw, list) and isinstance(proba_raw[0], dict):
        return np.array([p[1] for p in proba_raw])

    proba_raw = np.array(proba_raw)

    if proba_raw.ndim == 1:
        return proba_raw

    return proba_raw[:, 1]

# =====================================================
# UI
# =====================================================
st.set_page_config(layout="wide")
st.title("🫀 CVD Prediction – Single Instance XAI")

st.subheader("🧾 Input Data Pasien")

input_data = {}

for feat in FEATURE_NAMES:
    default_val = DEFAULT_VALUES.get(feat, 0.0)

    input_data[feat] = st.number_input(
        label=feat,
        value=float(default_val),
        step=1.0 if float(default_val).is_integer() else 0.1,
        format="%.2f"
    )

x_input = np.array([[input_data[f] for f in FEATURE_NAMES]])

# =====================================================
# PREDICTION
# =====================================================
prob = float(predict_proba(x_input)[0])
pred = int(prob >= 0.5)

st.subheader("📌 Hasil Prediksi")
st.metric("Probabilitas CVD", f"{prob:.4f}")
st.metric("Prediksi", "CVD" if pred else "No CVD")

# =====================================================
# LIME-LIKE LOCAL (PERTURBATION)
# =====================================================
st.subheader("🧩 LIME-like Local Explanation")

delta = 0.1
base_prob = prob
lime_scores = {}

for i, feat in enumerate(FEATURE_NAMES):
    x_p = x_input.copy()
    x_p[0, i] += delta
    new_prob = predict_proba(x_p)[0]
    lime_scores[feat] = abs(new_prob - base_prob)

lime_df = (
    pd.DataFrame({
        "Feature": lime_scores.keys(),
        "LIME_Local": lime_scores.values()
    })
    .sort_values("LIME_Local", ascending=False)
)

st.dataframe(lime_df, use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.barh(lime_df["Feature"], lime_df["LIME_Local"])
ax1.set_title("LIME-like Local Feature Importance")
ax1.invert_yaxis()
ax1.set_facecolor("white")
st.pyplot(fig1)

# =====================================================
# SHAP-LIKE GLOBAL (SAMPLING)
# =====================================================
st.subheader("📊 SHAP-like Global Explanation")

n_samples = 100
shap_scores = {f: [] for f in FEATURE_NAMES}

for _ in range(n_samples):
    noise = np.random.normal(0, 0.1, size=x_input.shape)
    x_s = x_input + noise
    base = predict_proba(x_s)[0]

    for i, feat in enumerate(FEATURE_NAMES):
        x_p = x_s.copy()
        x_p[0, i] += delta
        new_p = predict_proba(x_p)[0]
        shap_scores[feat].append(abs(new_p - base))

shap_df = (
    pd.DataFrame({
        "Feature": shap_scores.keys(),
        "SHAP_Global": [np.mean(v) for v in shap_scores.values()]
    })
    .sort_values("SHAP_Global", ascending=False)
)

st.dataframe(shap_df, use_container_width=True)

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

comp = shap_df.merge(lime_df, on="Feature")
comp["SHAP_Rank"] = comp["SHAP_Global"].rank(ascending=False)
comp["LIME_Rank"] = comp["LIME_Local"].rank(ascending=False)

rho, _ = spearmanr(comp["SHAP_Rank"], comp["LIME_Rank"])
tau, _ = kendalltau(comp["SHAP_Rank"], comp["LIME_Rank"])

c1, c2 = st.columns(2)
c1.metric("Spearman ρ", f"{rho:.3f}")
c2.metric("Kendall τ", f"{tau:.3f}")

st.dataframe(comp.sort_values("SHAP_Rank"), use_container_width=True)
