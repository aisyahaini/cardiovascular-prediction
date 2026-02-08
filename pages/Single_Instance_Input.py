import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("adaboost_model.onnx")  # ganti sesuai nama model kamu

model = load_model()

# =============================
# FEATURE NAMES (WAJIB SAMA DENGAN TRAINING)
# =============================
FEATURE_NAMES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "restecg",
    "max_heart_rate_achieved",
    "exercise_induced_angina",
    "st_depression",
    "st_slope_type",
    "num_major_vessels",
    "thalassemia_type"
]

# =============================
# PREDICT PROBA (AMAN LIST / ARRAY)
# =============================
def predict_proba(x):
    proba = model.predict_proba(x)

    # pastikan numpy array
    proba = np.asarray(proba)

    # ambil kelas positif
    if proba.ndim == 1:
        return proba
    return proba[:, 1]

# =============================
# STREAMLIT UI
# =============================
st.title("❤️ Cardiovascular Disease Prediction (Single Input)")

with st.form("input_form"):
    age = st.number_input("Usia", 1, 120, 45)
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0

    chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    resting_blood_pressure = st.number_input("Tekanan Darah Istirahat", 80, 250, 120)
    cholesterol = st.number_input("Kolesterol", 100, 600, 230)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    max_heart_rate_achieved = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", [0, 1])
    st_depression = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    st_slope_type = st.selectbox("ST Slope", [0, 1, 2])
    num_major_vessels = st.selectbox("Jumlah Pembuluh Darah (CA)", [0, 1, 2, 3])
    thalassemia_type = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Prediksi")

# =============================
# PROSES PREDIKSI
# =============================
if submit:
    input_dict = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "restecg": restecg,
        "max_heart_rate_achieved": max_heart_rate_achieved,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "st_slope_type": st_slope_type,
        "num_major_vessels": num_major_vessels,
        "thalassemia_type": thalassemia_type
    }

    # susun sesuai FEATURE_NAMES
    x_input = np.array([[input_dict[f] for f in FEATURE_NAMES]])

    base_prob = float(predict_proba(x_input)[0])
    prediction = int(base_prob >= 0.5)

    st.subheader("📊 Hasil Prediksi")
    st.write(f"**Probabilitas Penyakit Jantung:** {base_prob:.3f}")
    st.success("POSITIF" if prediction == 1 else "NEGATIF")

    # =====================================================
    # LIME-LIKE LOCAL FEATURE IMPORTANCE (FIXED)
    # =====================================================
    st.subheader("🧠 LIME-like Local Feature Importance")

    lime_scores = {}

    for i, feat in enumerate(FEATURE_NAMES):
        x_p = x_input.copy()

        # adaptive delta (KRUSIAL)
        delta = abs(x_input[0, i]) * 0.2 if x_input[0, i] != 0 else 1
        x_p[0, i] += delta

        new_prob = float(predict_proba(x_p)[0])
        lime_scores[feat] = abs(new_prob - base_prob)

    # sorting
    lime_scores = dict(sorted(lime_scores.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(lime_scores.keys()), list(lime_scores.values()))
    ax.set_title("LIME-like Local Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    # =====================================================
    # SHAP-LIKE GLOBAL EXPLANATION (MODEL-BASED)
    # =====================================================
    st.subheader("🌍 SHAP-like Global Explanation")

    if hasattr(model, "feature_importances_"):
        global_importance = model.feature_importances_
    else:
        global_importance = np.mean([
            est.feature_importances_
            for est in model.estimators_
        ], axis=0)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(FEATURE_NAMES, global_importance)
    ax2.set_title("SHAP-like Global Feature Importance")
    st.pyplot(fig2)

