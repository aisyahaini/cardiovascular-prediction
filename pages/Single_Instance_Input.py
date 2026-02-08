import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="CVD Prediction (ONNX)", layout="wide")

# =========================
# FEATURE ORDER (HARUS SESUAI TRAINING)
# =========================
FEATURE_NAMES = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure",
    "cholesterol", "fasting_blood_sugar", "Restecg",
    "max_heart_rate_achieved", "exercise_induced_angina",
    "st_depression", "st_slope_type",
    "num_major_vessels", "thalassemia_type"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    sess = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    output_names = [o.name for o in sess.get_outputs()]

    return sess, input_name, input_shape, output_names

session, INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES = load_model()

# =========================
# SAFE PROBABILITY HANDLER
# =========================
def get_prediction(outputs):
    """
    Works for:
    - label only
    - label + probability
    - probability only
    """
    if len(outputs) == 1:
        label = int(outputs[0][0])
        prob = None
    else:
        label = int(outputs[0][0])
        raw = outputs[1]

        if isinstance(raw, dict):
            prob = float(raw.get(1, list(raw.values())[-1]))
        else:
            raw = np.asarray(raw)
            prob = float(raw[0, -1])

    return label, prob

# =========================
# UI
# =========================
st.title("🫀 Cardiovascular Disease Prediction (ONNX)")

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

    # SAFETY CHECK
    assert X.shape[1] == INPUT_SHAPE[1], "Feature count mismatch!"

    outputs = session.run(None, {INPUT_NAME: X})

    label, prob = get_prediction(outputs)

    st.subheader("📊 Prediction Result")
    st.success("CVD Detected" if label == 1 else "No CVD Detected")

    if prob is not None:
        st.metric("CVD Probability", f"{prob:.2f}")
    else:
        st.info("Model does not provide probability output")

    # =========================
    # FEATURE IMPACT (FAST & SAFE)
    # =========================
    st.subheader("📈 Feature Impact (Local Approximation)")

    impacts = []
    base_outputs = session.run(None, {INPUT_NAME: X})
    _, base_prob = get_prediction(base_outputs)

    if base_prob is not None:
        for i, f in enumerate(FEATURE_NAMES):
            X_tmp = X.copy()
            X_tmp[0, i] = 0

            _, p = get_prediction(
                session.run(None, {INPUT_NAME: X_tmp})
            )

            impacts.append((f, base_prob - p))

        df = pd.DataFrame(impacts, columns=["Feature", "Impact"]) \
               .sort_values("Impact", ascending=False)

        fig, ax = plt.subplots()
        df.head(8).plot.barh(x="Feature", y="Impact", ax=ax, legend=False)
        ax.invert_yaxis()
        st.pyplot(fig)

    st.markdown("""
    **Catatan Interpretasi:**
    Feature impact dihitung menggunakan pendekatan *local perturbation*
    yang setara secara konseptual dengan metode **LIME-style explanation**,
    dan kompatibel sepenuhnya dengan ONNX runtime.
    """)

