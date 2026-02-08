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
# SAFE PREDICT PROBA
# =====================================================
def predict_proba(x):
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: x.astype(np.float32)}
    )

    proba_raw = outputs[1]

    # case: list of dict (sklearn-style ONNX)
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
st.title("🫀 Prediksi Manual Penyakit Cardiovascular (Single Input)")

# =====================================================
# INPUT FORM
# =====================================================
with st.form("input_form"):
    st.subheader("🧾 Input Data Pasien")

    age = st.number_input("Usia", 1, 120, 45)

    sex_label = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex_label == "Laki-laki" else 0

    country = st.selectbox("Dataset / Country", [0, 1, 2, 3])

    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 250, 120)
    chol = st.number_input("Kolesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah (CA)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================================
# MAIN LOGIC
# =====================================================
if submit:

    # =====================
    # BUILD INPUT DICT (NAMA HARUS SAMA DENGAN MODEL)
    # =====================
    input_dict = {
        "age": age,
        "sex": sex,
        "country": country,   # ✅ FIX UTAMA
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # =====================
    # ALIGN KE FEATURE_NAMES (ANTI KeyError)
    # =====================
    x_input = np.array([[input_dict.get(f, 0) for f in FEATURE_NAMES]])

    # =====================
    # PREDICTION
    # =====================
    prob = float(predict_proba(x_input)[0])
    pred = int(prob >= 0.5)

    st.subheader("📌 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.4f}")
    st.metric("Prediksi", "CVD" if pred else "No CVD")

    # =====================================================
    # LIME-LIKE LOCAL EXPLANATION (FIXED)
    # =====================================================
    st.subheader("🧩 LIME-Local Explanation")
    
    base_prob = prob
    lime_scores = {}
    
    for i, feat in enumerate(FEATURE_NAMES):
        x_p = x_input.copy()
    
        val = x_input[0, i]
    
        # ✅ ADAPTIVE DELTA (KRUSIAL)
        delta = max(1.0, abs(val) * 0.2)
    
        x_p[0, i] += delta
        new_prob = float(predict_proba(x_p)[0])
    
        lime_scores[feat] = abs(new_prob - base_prob)
    
    lime_df = (
        pd.DataFrame({
            "Feature": lime_scores.keys(),
            "LIME_Local": lime_scores.values()
        })
        .sort_values("LIME_Local", ascending=False)
    )
    
    st.dataframe(lime_df)
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.barh(lime_df["Feature"], lime_df["LIME_Local"])
    ax1.set_title("LIME-like Local Feature Importance")
    ax1.invert_yaxis()
    st.pyplot(fig1)


    # =====================================================
    # SHAP-LIKE GLOBAL EXPLANATION (FIXED)
    # =====================================================
    st.subheader("📊 SHAP – Local Explanation")
    
    n_samples = 50
    shap_scores = {f: [] for f in FEATURE_NAMES}
    
    for _ in range(n_samples):
        noise = np.random.normal(0, 1.0, size=x_input.shape)
        x_s = x_input + noise
    
        base = float(predict_proba(x_s)[0])
    
        for i, feat in enumerate(FEATURE_NAMES):
            x_p = x_s.copy()
    
            val = x_s[0, i]
            delta = max(1.0, abs(val) * 0.2)
    
            x_p[0, i] += delta
            new_p = float(predict_proba(x_p)[0])
    
            shap_scores[feat].append(abs(new_p - base))
    
    shap_df = (
        pd.DataFrame({
            "Feature": shap_scores.keys(),
            "SHAP_Global": [np.mean(v) for v in shap_scores.values()]
        })
        .sort_values("SHAP_Global", ascending=False)
    )
    
    st.dataframe(shap_df)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.barh(shap_df["Feature"], shap_df["SHAP_Global"])
    ax2.set_title("SHAP – Local Explanation")
    ax2.invert_yaxis()
    st.pyplot(fig2)


    # =====================
    # KESIMPULAN ILMIAH
    # =====================
    st.subheader("📌 Kesimpulan Ilmiah")
    
    # =====================================================
    # CONSISTENCY ANALYSIS
    # =====================================================
    st.markdown("📐 Konsistensi SHAP-like vs LIME-like")

    comp = shap_df.merge(lime_df, on="Feature")
    comp["SHAP_Rank"] = comp["SHAP_Global"].rank(ascending=False)
    comp["LIME_Rank"] = comp["LIME_Local"].rank(ascending=False)

    rho, _ = spearmanr(comp["SHAP_Rank"], comp["LIME_Rank"])
    tau, _ = kendalltau(comp["SHAP_Rank"], comp["LIME_Rank"])

    c1, c2 = st.columns(2)
    c1.metric("Spearman ρ", f"{rho:.3f}")
    c2.metric("Kendall τ", f"{tau:.3f}")

    st.dataframe(comp.sort_values("SHAP_Rank"))

    # =====================
    # KESIMPULAN ILMIAH
    # =====================
    st.subheader("📌 Kesimpulan Ilmiah")

    st.markdown("""
    - **LIME lebih unggul untuk analisis individual**, karena menjelaskan keputusan model
      secara spesifik pada satu pasien.
    - **SHAP tetap memiliki keunggulan**, terutama untuk analisis global karena konsisten
      secara teoritis dan stabil terhadap seluruh dataset.
    - Oleh karena itu, pada **single input**, LIME menjadi metode utama,
      sementara SHAP berfungsi sebagai pendukung interpretasi global model.
    """)



