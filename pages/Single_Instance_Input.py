import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import lime.lime_tabular
import json

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Prediksi Cardiovascular – Single Input (ONNX)",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_onnx():
    sess = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    return sess, feature_names, input_name, output_names


session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_onnx()

# =====================================================
# UTIL: SAFE ONNX PROBA
# =====================================================
def onnx_predict_proba(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)  # 🔥 FIX UTAMA

    outputs = session.run(
        None,  # 🔥 JANGAN PAKSA OUTPUT_NAMES
        {INPUT_NAME: X}
    )

    # Umumnya output[1] = probability
    proba_raw = outputs[-1]

    probs = []
    for p in proba_raw:
        if isinstance(p, dict):
            probs.append([p.get(0, 0.0), p.get(1, 0.0)])
        else:
            probs.append(p)

    return np.array(probs, dtype=np.float32)

# =====================================================
# UI
# =====================================================
st.title("🫀 Prediksi Penyakit Cardiovascular – Single Input (ONNX)")

with st.form("form"):
    age = st.number_input("Usia", 1, 120, 45)
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0
    dataset = st.selectbox("Dataset", [0, 1, 2, 3])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah", 80, 250, 120)
    chol = st.number_input("Kolesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================================
# INFERENCE
# =====================================================
if submit:

    raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "country": dataset,
        "chest_pain_type": cp,
        "resting_blood_pressure": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": fbs,
        "Restecg": restecg,
        "max_heart_rate_achieved": thalch,
        "exercise_induced_angina": exang,
        "st_depression": oldpeak,
        "st_slope_type": slope,
        "num_major_vessels": ca,
        "thalassemia_type": thal
    }])

    # Samakan fitur
    for f in FEATURE_NAMES:
        if f not in raw.columns:
            raw[f] = 0

    X = raw[FEATURE_NAMES].astype(np.float32).values  # 🔥 FIX UTAMA

    probs = onnx_predict_proba(X)
    prob = float(probs[0, 1])
    pred = int(prob >= 0.5)

    st.subheader("📌 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.write("Prediksi:", "🟥 CVD" if pred else "🟩 Tidak CVD")

    # =====================================================
    # LOCAL FEATURE IMPACT (AMAN)
    # =====================================================
    # =====================
# LIME (ONNX SAFE VERSION)
# =====================
st.subheader("🧩 LIME – Local Explanation")

# ---- Wrapper predict_proba untuk ONNX ----
def onnx_predict_proba(x):
    x = np.asarray(x, dtype=np.float32)
    out = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: x}
    )

    # Pastikan output probabilitas 2D
    proba = out[1]
    if proba.ndim == 1:
        proba = np.vstack([1 - proba, proba]).T

    return proba


# ---- Background data (WAJIB: BUKAN ZEROS) ----
rng = np.random.default_rng(42)
background = rng.normal(
    loc=input_df.values,
    scale=0.01,
    size=(50, input_df.shape[1])
).astype(np.float32)

# ---- LIME Explainer (TANPA forward_selection) ----
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=background,
    feature_names=FEATURE_NAMES,
    class_names=["No CVD", "CVD"],
    mode="classification",
    discretize_continuous=False,
    feature_selection="none"  # 🔥 FIX UTAMA
)

# ---- Explain instance ----
exp = explainer.explain_instance(
    input_df.values[0],
    onnx_predict_proba,
    num_features=5
)

# ---- Plot ----
fig = exp.as_pyplot_figure()
fig.patch.set_facecolor("white")
st.pyplot(fig)

