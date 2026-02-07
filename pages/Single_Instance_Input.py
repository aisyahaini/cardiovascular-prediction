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
# UTIL: AMAN EKSTRAK PROBABILITAS ONNX
# =====================================================
def extract_positive_proba(proba_output):
    """
    Aman untuk semua format output ONNX:
    - numpy array
    - list
    - list of dict (sklearn -> ONNX)
    """
    # List (umumnya list of dict)
    if isinstance(proba_output, list):
        first = proba_output[0]

        if isinstance(first, dict):
            return float(first.get(1, 0.0))

        return float(first[1])

    # Numpy array
    if isinstance(proba_output, np.ndarray):
        if proba_output.ndim == 2:
            return float(proba_output[0, 1])
        return float(proba_output[0])

    return float(proba_output)

# =====================================================
# LOAD ONNX MODEL
# =====================================================
@st.cache_resource
def load_onnx():
    session = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    return session, feature_names, input_name, output_names


session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_onnx()

# =====================================================
# UI
# =====================================================
st.title("🫀 Prediksi Penyakit Cardiovascular – Single Input (ONNX)")

with st.form("input_form"):
    age = st.number_input("Usia", 1, 120, 45)
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0

    dataset = st.selectbox("Dataset", [0, 1, 2, 3])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 250, 120)
    chol = st.number_input("Kolesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah (CA)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================================
# INFERENCE
# =====================================================
if submit:

    # -----------------
    # INPUT KE FORMAT MODEL
    # -----------------
    input_df = pd.DataFrame([{
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
    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FEATURE_NAMES].astype(np.float32)

    # -----------------
    # ONNX PREDICT
    # -----------------
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: input_df.values}
    )

    pred = int(outputs[0][0])
    prob = extract_positive_proba(outputs[1])

    # -----------------
    # OUTPUT
    # -----------------
    st.subheader("📌 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.write("Prediksi:", "🟥 CVD" if pred == 1 else "🟩 Tidak CVD")

    # =====================================================
    # LOCAL FEATURE IMPACT (AMAN ONNX)
    # =====================================================
    st.subheader("🧠 Local Feature Impact")

    impacts = []
    base_prob = prob

    for col in FEATURE_NAMES:
        temp = input_df.copy()
        temp[col] = 0

        out = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: temp.values}
        )

        new_prob = extract_positive_proba(out[1])
        impacts.append([col, base_prob - new_prob])

    importance_df = pd.DataFrame(
        impacts, columns=["Feature", "Impact"]
    ).sort_values("Impact", ascending=False)

    st.dataframe(importance_df.head(8))

    fig, ax = plt.subplots()
    importance_df.head(8).plot.barh(
        x="Feature",
        y="Impact",
        ax=ax,
        legend=False
    )
    ax.invert_yaxis()
    plt.title("Local Feature Impact")
    st.pyplot(fig)

    # =====================================================
    # LIME (ONNX WRAPPER – STABLE)
    # =====================================================
    st.subheader("🧩 LIME – Local Explanation")

    def onnx_predict_proba(x):
        out = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: x.astype(np.float32)}
        )

        proba_out = out[1]
        probs = []

        for p in proba_out:
            if isinstance(p, dict):
                probs.append([p.get(0, 0.0), p.get(1, 0.0)])
            else:
                probs.append(p)

        return np.array(probs, dtype=np.float32)

    background = np.zeros((20, len(FEATURE_NAMES)), dtype=np.float32)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background,
        feature_names=FEATURE_NAMES,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        input_df.iloc[0].values,
        onnx_predict_proba,
        num_features=5
    )

    st.pyplot(exp.as_pyplot_figure())

    # =====================================================
    # INTERPRETASI
    # =====================================================
    st.subheader("📝 Interpretasi Prediksi")

    confidence = (
        "tinggi" if prob >= 0.75 else
        "sedang" if prob >= 0.5 else
        "rendah"
    )

    st.markdown(f"""
    Model **AdaBoost (ONNX)** memprediksi risiko penyakit kardiovaskular
    dengan tingkat keyakinan **{confidence}**
    (probabilitas **{prob:.2f}**).
    """)
