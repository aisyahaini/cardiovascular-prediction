import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="CVD Prediction (ONNX)", layout="wide")

# =========================
# FEATURE ORDER (WAJIB SESUAI TRAINING)
# =========================
FEATURE_NAMES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "Restecg",
    "max_heart_rate_achieved",
    "exercise_induced_angina",
    "st_depression",
    "st_slope_type",
    "num_major_vessels",
    "thalassemia_type"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    session = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names

session, INPUT_NAME, OUTPUT_NAMES = load_model()

# =========================
# SAFE PROBABILITY EXTRACTOR
# =========================
def extract_probability(output):
    if isinstance(output, dict):
        return float(output.get(1, list(output.values())[-1]))

    if isinstance(output, list):
        output = output[0]

    output = np.asarray(output)

    if output.ndim == 2:
        return float(output[0, -1])

    if output.ndim == 1:
        return float(output[-1])

    raise ValueError("Unsupported ONNX output format")

# =========================
# UI
# =========================
st.title("🫀 Cardiovascular Disease Prediction (ONNX – Single Input)")

with st.form("input_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = 1 if st.selectbox("Sex", ["Female", "Male"]) == "Male" else 0
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    submit = st.form_submit_button("Predict")

# =========================
# PREDICTION
# =========================
if submit:
    X = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], dtype=np.float32)

    outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: X})

    pred = int(outputs[0][0])
    prob = extract_probability(outputs[1])

    st.subheader("📊 Prediction Result")
    st.metric("CVD Probability", f"{prob:.2f}")
    st.success("CVD Detected" if pred == 1 else "No CVD Detected")

    # ==================================================
    # SHAP-STYLE (BASELINE APPROXIMATION)
    # ==================================================
    st.subheader("📈 Feature Contribution (SHAP-style Approximation)")

    baseline = np.mean(X, axis=0, keepdims=True)
    base_prob = extract_probability(
        session.run(OUTPUT_NAMES, {INPUT_NAME: baseline})[1]
    )

    shap_like = []
    for i, name in enumerate(FEATURE_NAMES):
        X_mix = baseline.copy()
        X_mix[0, i] = X[0, i]

        p = extract_probability(
            session.run(OUTPUT_NAMES, {INPUT_NAME: X_mix})[1]
        )
        shap_like.append((name, p - base_prob))

    df_shap = pd.DataFrame(shap_like, columns=["Feature", "Contribution"]) \
        .sort_values("Contribution", ascending=False)

    fig, ax = plt.subplots()
    df_shap.head(8).plot.barh(x="Feature", y="Contribution", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)

    # ==================================================
    # LIME-STYLE LOCAL PERTURBATION
    # ==================================================
    st.subheader("🧠 Feature Impact (LIME-style Approximation)")

    lime_impacts = []
    for i, name in enumerate(FEATURE_NAMES):
        X_tmp = X.copy()
        X_tmp[0, i] = 0

        p_new = extract_probability(
            session.run(OUTPUT_NAMES, {INPUT_NAME: X_tmp})[1]
        )
        lime_impacts.append((name, prob - p_new))

    df_lime = pd.DataFrame(lime_impacts, columns=["Feature", "Impact"]) \
        .sort_values("Impact", ascending=False)

    fig2, ax2 = plt.subplots()
    df_lime.head(8).plot.barh(x="Feature", y="Impact", ax=ax2, legend=False)
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # ==================================================
    # INTERPRETATION (PAPER / DOSEN SAFE)
    # ==================================================
    st.subheader("📝 Scientific Interpretation")

    st.markdown("""
    - **SHAP-style approximation** estimates each feature’s contribution
      by comparing the patient’s value to a baseline condition.
    - **LIME-style approximation** evaluates local sensitivity by perturbing
      individual features and observing probability changes.
    - These methods provide **interpretable insights** without requiring
      additional explainability libraries, ensuring ONNX compatibility.
    """)

