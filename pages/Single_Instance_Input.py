import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# =====================
# PAGE CONFIG (WAJIB PALING ATAS)
# =====================
st.set_page_config(
    page_title="Prediksi Cardiovascular – Single Input",
    layout="wide"
)

# =====================
# LOAD MODEL (AMAN)
# =====================
@st.cache_resource
def load_model():
    return joblib.load("adaboost_modelfix.pkl")

model = load_model()
feature_names = model.feature_names_in_

st.title("🫀 Prediksi Penyakit Cardiovascular – Single Input")

# =====================
# FORM INPUT
# =====================
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

# =====================
# PREDIKSI
# =====================
if submit:
    input_df = pd.DataFrame([[ 
        age, sex, dataset, cp, trestbps, chol, fbs,
        restecg, thalch, exang, oldpeak, slope, ca, thal
    ]], columns=[
        "age","sex","dataset","cp","trestbps","chol","fbs",
        "restecg","thalch","exang","oldpeak","slope","ca","thal"
    ])

    # Mapping fitur
    mapping = {
        "ca": "num_major_vessels",
        "cp": "chest_pain_type",
        "thal": "thalassemia_type",
        "slope": "st_slope_type",
        "chol": "cholesterol",
        "oldpeak": "st_depression",
        "trestbps": "resting_blood_pressure",
        "sex": "sex",
        "age": "age",
        "thalch": "max_heart_rate_achieved",
        "restecg": "Restecg",
        "exang": "exercise_induced_angina",
        "fbs": "fasting_blood_sugar",
        "dataset": "country"
    }

    input_df = input_df.rename(columns=mapping)

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # =====================
    # HASIL PREDIKSI
    # =====================
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.write("Prediksi:", "🟥 CVD" if pred == 1 else "🟩 Tidak CVD")

    # =====================
    # SHAP (TREE EXPLAINER – AMAN)
    # =====================
    st.subheader("🧠 SHAP – Local Explanation")

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        input_df,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)

    st.caption(
        "SHAP menggunakan TreeExplainer yang efisien dan stabil "
        "untuk model berbasis tree seperti AdaBoost."
    )

    # =====================
    # LIME (OPSIONAL & AMAN)
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    import lime.lime_tabular

    background = np.zeros((10, input_df.shape[1]))

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        input_df.iloc[0].values,
        model.predict_proba,
        num_features=5
    )

    fig_lime = exp.as_pyplot_figure(label=1)
    fig_lime.patch.set_facecolor("white")
    st.pyplot(fig_lime)

    # =====================
    # INTERPRETASI OTOMATIS
    # =====================
    st.subheader("📝 Interpretasi Prediksi")

    lime_results = exp.as_list(label=1)

    positive = [f for f in lime_results if f[1] > 0]
    negative = [f for f in lime_results if f[1] <= 0]

    confidence = "tinggi" if prob >= 0.75 else "sedang" if prob >= 0.5 else "rendah"

    st.markdown(f"""
    Model **AdaBoost** memprediksi risiko penyakit kardiovaskular
    dengan tingkat keyakinan **{confidence}** (probabilitas **{prob:.2f}**).
    """)

    if positive:
        st.markdown("🔺 **Fitur yang meningkatkan risiko:**")
        for f, w in positive:
            st.markdown(f"- {f} (kontribusi: {w:.3f})")

    if negative:
        st.markdown("🔻 **Fitur yang menurunkan risiko:**")
        for f, w in negative:
            st.markdown(f"- {f} (kontribusi: {w:.3f})")

    # =====================
    # KESIMPULAN
    # =====================
    st.subheader("📌 Kesimpulan Ilmiah")

    st.markdown("""
    - **SHAP TreeExplainer** digunakan karena efisien dan stabil
      untuk model berbasis pohon.
    - **LIME** memberikan interpretasi lokal yang intuitif
      pada satu pasien.
    - Kombinasi SHAP dan LIME meningkatkan transparansi model
      tanpa mengorbankan performa sistem.
    """)



