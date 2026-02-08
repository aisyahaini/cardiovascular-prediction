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
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    sess = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("feature_names.json") as f:
        feature_names = json.load(f)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    return sess, feature_names, input_name, output_names


session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_model()

# =========================
# SAFE PROBA EXTRACTOR
# =========================
def extract_probability(raw):
    """
    Universal ONNX probability extractor
    Works for:
    - numpy array
    - list
    - dict
    """
    # Case 1: dict (sklearn AdaBoost often)
    if isinstance(raw, dict):
        return float(raw.get(1, list(raw.values())[-1]))

    # Case 2: list of dict
    if isinstance(raw, list) and isinstance(raw[0], dict):
        return float(raw[0].get(1, list(raw[0].values())[-1]))

    # Case 3: numpy array
    raw = np.asarray(raw)

    if raw.ndim == 2:
        return float(raw[0, 1])

    if raw.ndim == 1:
        return float(raw[0])

    raise ValueError("Unknown ONNX probability output format")

# =========================
# UI
# =========================
st.title("🫀 Cardiovascular Disease Prediction (ONNX)")

with st.form("input_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0
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

    X = (
        input_data[FEATURE_NAMES]
        .astype(np.float32)
        .values
    )

    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: X}
    )

    # LABEL
    pred = int(outputs[0][0])

    # PROBABILITY (SAFE)
    prob = extract_probability(outputs[1])

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Result")
    st.metric("CVD Probability", f"{prob:.2f}")

    st.success("CVD Detected" if pred == 1 else "No CVD Detected")

    # =========================
    # SIMPLE FEATURE IMPACT
    # =========================
    st.subheader("🧠 Feature Impact")

    impacts = []
    base_prob = prob

    for i, name in enumerate(FEATURE_NAMES):
        X_temp = X.copy()
        X_temp[0, i] = 0

        out = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: X_temp}
        )

        new_prob = extract_probability(out[1])
        impacts.append((name, base_prob - new_prob))

    df_imp = pd.DataFrame(impacts, columns=["Feature", "Impact"]) \
        .sort_values("Impact", ascending=False)

    st.dataframe(df_imp.head(8))

    fig, ax = plt.subplots()
    df_imp.head(8).plot.barh(x="Feature", y="Impact", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)
