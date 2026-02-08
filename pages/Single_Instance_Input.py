import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="CVD Prediction (ONNX)", layout="wide")

# =========================
# FEATURE ORDER (MANUAL & FIX)
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

    # Dynamic shape safety
    if isinstance(input_dim, str) or input_dim is None:
        input_dim = len(FEATURES)

    return sess, input_name, input_dim

session, INPUT_NAME, MODEL_DIM = load_model()

# =========================
# SUPER SAFE OUTPUT PARSER
# =========================
def parse_output(outputs):
    """
    Parse ONNX outputs safely.
    Works even if model outputs ONLY label.
    """
    # ---------- LABEL ----------
    label_raw = outputs[0]
    label = int(np.asarray(label_raw).ravel()[0])

    # ---------- PROBABILITY ----------
    prob = None

    if len(outputs) > 1:
        raw = outputs[1]

        try:
            # dict output (rare but possible)
            if isinstance(raw, dict):
                vals = list(raw.values())
                if len(vals) > 0 and isinstance(vals[-1], (float, int)):
                    prob = float(vals[-1])

            # numpy / list
            elif raw is not None:
                arr = np.asarray(raw).astype(np.float32)

                if arr.size > 0 and np.isfinite(arr.ravel()[0]):
                    prob = float(arr.ravel()[0])

        except Exception:
            prob = None

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

    # Match ONNX input dimension
    X = np.zeros((1, MODEL_DIM), dtype=np.float32)
    X[0, :min(len(raw_input), MODEL_DIM)] = raw_input[:MODEL_DIM]

    outputs = session.run(None, {INPUT_NAME: X})
    label, prob = parse_output(outputs)

    # =========================
    # RESULT
    # =========================
    st.subheader("📊 Prediction Result")

    st.success("CVD Detected" if label == 1 else "No CVD Detected")

    if prob is not None:
        st.metric("Estimated Risk Score", f"{prob:.2f}")
    else:
        st.info("Model does not provide probability output (label-only ONNX model)")

    # =========================
    # FEATURE IMPACT (FAST & SAFE)
    # =========================
    if prob is not None:
        st.subheader("📈 Local Feature Impact (LIME-style Approximation)")

        impacts = []

        for i, name in enumerate(FEATURES):
            if i >= MODEL_DIM:
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
    **Catatan Ilmiah**  
    Interpretasi dilakukan menggunakan *local perturbation analysis*  
    yang secara konseptual setara dengan **LIME** dan kompatibel penuh  
    dengan model **ONNX tanpa SHAP/LIME library**.
    """)
