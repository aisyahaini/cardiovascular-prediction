import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import lime.lime_tabular
import json

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Prediksi Cardiovascular (ONNX)",
    layout="wide"
)

# =====================
# LOAD ONNX MODEL
# =====================
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

# =====================
# TITLE
# =====================
st.title("🫀 Prediksi Penyakit Cardiovascular (Single Input – ONNX)")

# =====================
# INPUT FORM
# =====================
with st.form("input_form"):
    age = st.number_input("Usia", 1, 120, 45)
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0

    country = st.selectbox("Dataset", [0, 1, 2, 3])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", 80, 250, 120)
    chol = st.number_input("Kolesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah (CA)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi")

# =====================
# PREDICTION
# =====================
if submit:

    # =====================
    # BUILD INPUT DATAFRAME
    # =====================
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "country": country,
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

    # Pastikan urutan & tipe fitur
    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0

    X = (
        input_df[FEATURE_NAMES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.float32)
        .values
    )

    # =====================
    # ONNX INFERENCE
    # =====================
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: X}
    )

    # ---- Prediksi label ----
    pred = int(outputs[0][0])

    # ---- Probabilitas (AMAN SEMUA FORMAT) ----
    proba_out = outputs[1]

    if proba_out.ndim == 2:
        prob = float(proba_out[0, 1])
    else:
        prob = float(proba_out[0])

    # =====================
    # OUTPUT
    # =====================
    st.subheader("📌 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.write("Prediksi:", "🟥 CVD" if pred == 1 else "🟩 Tidak CVD")

    # =====================
    # LOCAL FEATURE IMPACT (AMAN, TANPA LOOP BERBAHAYA)
    # =====================
    st.subheader("🧠 Local Feature Impact")

    base_prob = prob
    impacts = []

    for i, col in enumerate(FEATURE_NAMES):
        X_temp = X.copy()
        X_temp[0, i] = 0

        out = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: X_temp}
        )

        proba_temp = out[1]
        new_prob = (
            float(proba_temp[0, 1])
            if proba_temp.ndim == 2
            else float(proba_temp[0])
        )

        impacts.append([col, base_prob - new_prob])

    imp_df = pd.DataFrame(
        impacts,
        columns=["Feature", "Impact"]
    ).sort_values("Impact", ascending=False)

    st.dataframe(imp_df.head(8))

    fig, ax = plt.subplots()
    imp_df.head(8).plot.barh(
        x="Feature",
        y="Impact",
        ax=ax,
        legend=False
    )
    ax.invert_yaxis()
    plt.title("Local Feature Impact")
    st.pyplot(fig)

    # =====================
    # LIME (ONNX SAFE – FINAL)
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    def onnx_predict_proba(x):
        x = np.asarray(x, dtype=np.float32)
        out = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: x}
        )

        proba = out[1]
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T

        return proba

    rng = np.random.default_rng(42)
    background = rng.normal(
        loc=X,
        scale=0.01,
        size=(50, X.shape[1])
    ).astype(np.float32)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background,
        feature_names=FEATURE_NAMES,
        class_names=["No CVD", "CVD"],
        mode="classification",
        discretize_continuous=False,
        feature_selection="none"
    )

    exp = explainer.explain_instance(
        X[0],
        onnx_predict_proba,
        num_features=5
    )

    fig_lime = exp.as_pyplot_figure()
    fig_lime.patch.set_facecolor("white")
    st.pyplot(fig_lime)

    # =====================
    # INTERPRETASI
    # =====================
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
