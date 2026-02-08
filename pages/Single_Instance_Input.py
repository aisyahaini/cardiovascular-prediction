import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import json
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="CVD Prediction (ONNX)", layout="wide")

# =========================
# LOAD MODEL & BASELINE
# =========================
@st.cache_resource
def load_model():
    sess = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("feature_names.json") as f:
        feature_names = json.load(f)

    with open("feature_baseline.json") as f:
        baseline = json.load(f)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    return sess, feature_names, baseline, input_name, output_names


session, FEATURE_NAMES, BASELINE, INPUT_NAME, OUTPUT_NAMES = load_model()

# =========================
# SAFE PROBA EXTRACTOR
# =========================
def extract_probability(raw):
    if isinstance(raw, dict):
        return float(raw.get(1, list(raw.values())[-1]))

    if isinstance(raw, list) and isinstance(raw[0], dict):
        return float(raw[0].get(1, list(raw[0].values())[-1]))

    raw = np.asarray(raw)

    if raw.ndim == 2:
        return float(raw[0, 1])

    if raw.ndim == 1:
        return float(raw[0])

    raise ValueError("Unknown ONNX probability output format")

# =========================
# UI
# =========================
st.title("🫀 Prediksi Penyakit Cardiovascular (Single Input – ONNX)")

with st.form("input_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = 1 if st.selectbox("Sex", ["Female", "Male"]) == "Male" else 0
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("Predict")

# =========================
# PREDICTION
# =========================
if submit:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest_pain_type": cp,
        "resting_blood_pressure": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": fbs,
        "Restecg": restecg,
        "max_heart_rate_achieved": thalach,
        "exercise_induced_angina": exang,
        "st_depression": oldpeak,
        "st_slope_type": slope,
        "num_major_vessels": ca,
        "thalassemia_type": thal
    }])

    for col in FEATURE_NAMES:
        if col not in input_data:
            input_data[col] = 0

    X = input_data[FEATURE_NAMES].astype(np.float32).values

    outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: X})
    pred = int(outputs[0][0])
    prob = extract_probability(outputs[1])

    st.subheader("📊 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.success("CVD Detected" if pred == 1 else "No CVD Detected")

    # ==================================================
    # SHAP-STYLE APPROXIMATION (BASELINE COMPARISON)
    # ==================================================
    st.subheader("📈 SHAP-style Feature Contribution (Approximation)")

    baseline_df = pd.DataFrame([BASELINE])[FEATURE_NAMES].astype(np.float32)
    X_base = baseline_df.values

    base_out = session.run(OUTPUT_NAMES, {INPUT_NAME: X_base})
    base_prob = extract_probability(base_out[1])

    shap_like = []

    for i, name in enumerate(FEATURE_NAMES):
        X_mix = X_base.copy()
        X_mix[0, i] = X[0, i]

        out = session.run(OUTPUT_NAMES, {INPUT_NAME: X_mix})
        p = extract_probability(out[1])

        shap_like.append((name, p - base_prob))

    df_shap = pd.DataFrame(shap_like, columns=["Feature", "SHAP_like"]) \
        .sort_values("SHAP_like", ascending=False)

    fig, ax = plt.subplots()
    df_shap.head(8).plot.barh(x="Feature", y="SHAP_like", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)

    # ==================================================
    # LIME-STYLE LOCAL PERTURBATION
    # ==================================================
    st.subheader("🧠 LIME-style Local Feature Impact")

    lime_impacts = []

    for i, name in enumerate(FEATURE_NAMES):
        X_temp = X.copy()
        X_temp[0, i] = 0

        out = session.run(OUTPUT_NAMES, {INPUT_NAME: X_temp})
        new_prob = extract_probability(out[1])

        lime_impacts.append((name, prob - new_prob))

    df_lime = pd.DataFrame(lime_impacts, columns=["Feature", "Impact"]) \
        .sort_values("Impact", ascending=False)

    fig2, ax2 = plt.subplots()
    df_lime.head(8).plot.barh(x="Feature", y="Impact", ax=ax2, legend=False)
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # ==================================================
    # INTERPRETASI
    # ==================================================
    st.subheader("📝 Interpretasi Ilmiah")

    st.markdown(f"""
    - Analisis **SHAP-style** menunjukkan bagaimana nilai fitur pasien
      **menyimpang dari kondisi populasi rata-rata**.
    - Analisis **LIME-style** menjelaskan **sensitivitas prediksi lokal**
      terhadap perubahan kecil pada fitur pasien.
    - Kombinasi keduanya memberikan interpretasi **global-context + local-decision**.
    """)
