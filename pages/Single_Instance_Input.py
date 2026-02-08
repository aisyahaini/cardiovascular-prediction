import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="CVD Prediction (ONNX)", layout="wide")

# =========================
# RAW FEATURES (ORDERED)
# =========================
FEATURES = [
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

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_dim = input_meta.shape[1]

    if isinstance(input_dim, str) or input_dim is None:
        input_dim = None

    return sess, input_name, input_dim

session, INPUT_NAME, MODEL_DIM = load_model()

# =========================
# ULTRA SAFE OUTPUT PARSER
# =========================
def parse_output(outputs):
    """
    Guaranteed-safe ONNX output parser
    """
    # Label (always first output)
    label = int(np.asarray(outputs[0]).ravel()[0])

    prob = None

    # Probability (if exists)
    if len(outputs) > 1:
        raw = outputs[1]

        if isinstance(raw, dict):
            prob = float(list(raw.values())[-1])

        else:
            raw = np.asarray(raw)

            if raw.size > 0:
                prob = float(raw.ravel()[0])

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
    raw_input = np.array([
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ], dtype=np.float32)

    # Match ONNX input shape
    if MODEL_DIM is None:
        X = raw_input.reshape(1, -1)
    else:
        X = np.zeros((1, MODEL_DIM), dtype=np.float32)
        X[0, :min(len(raw_input), MODEL_DIM)] = raw_input[:MODEL_DIM]

    outputs = session.run(None, {INPUT_NAME: X})
    label, prob = parse_output(outputs)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Prediction Result")

    st.success("CVD Detected" if label == 1 else "No CVD Detected")

    if prob is not None:
        st.metric("CVD Probability", f"{prob:.2f}")
    else:
        st.info("Model does not provide probability output")

    # =========================
    # FEATURE IMPACT (FAST)
    # =========================
    if prob is not None:
        st.subheader("📈 Feature Impact (Local Approximation)")

        impacts = []

        for i, name in enumerate(FEATURES):
            if i >= X.shape[1]:
                continue

            X_tmp = X.copy()
            X_tmp[0, i] = 0

            _, p2 = parse_output(
                session.run(None, {INPUT_NAME: X_tmp})
            )

            if p2 is not None:
                impacts.append((name, prob - p2))

        if impacts:
            df = pd.DataFrame(
                impacts, columns=["Feature", "Impact"]
            ).sort_values("Impact", ascending=False)

            fig, ax = plt.subplots()
            df.plot.barh(x="Feature", y="Impact", ax=ax, legend=False)
            ax.invert_yaxis()
            st.pyplot(fig)

    st.markdown("""
    **Catatan Interpretasi**  
    Feature impact dihitung menggunakan pendekatan *local perturbation*  
    yang secara konseptual setara dengan **LIME-style explanation**,  
    dan sepenuhnya kompatibel dengan **ONNX Runtime**.
    """)
