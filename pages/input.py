import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    return joblib.load("adaboost_modelfix.pkl")

model = load_model()
feature_names = model.feature_names_in_

st.title("✍️ Prediksi Manual Penyakit Cardiovascular")

# =====================
# INPUT FORM
# =====================
with st.form("input_form"):
    age = st.number_input("Usia", 1, 120, 45)
    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0

    dataset = st.selectbox("Dataset", [0, 1, 2, 3])
    cp = st.selectbox("Chest Pain", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah", 80, 250, 120)
    chol = st.number_input("Kolesterol", 100, 600, 230)
    fbs = st.selectbox("FBS > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    submit = st.form_submit_button("Prediksi")

    df_to_model_mapping = {
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


# =====================
# PREDIKSI
# =====================
if submit:
    input_data = pd.DataFrame([[
        age, sex, dataset, cp, trestbps, chol, fbs,
        restecg, thalch, exang, oldpeak, slope, ca, thal
    ]], columns=[
        "age","sex","dataset","cp","trestbps","chol","fbs",
        "restecg","thalch","exang","oldpeak","slope","ca","thal"
    ])

    # rename kolom input manual → nama fitur model
    input_data = input_data.rename(columns=df_to_model_mapping)

    # pastikan semua fitur model ada
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # urutkan sesuai model
    input_data = input_data[model.feature_names_in_]


    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Hasil Prediksi")
    st.metric("Probabilitas CVD", f"{prob:.2f}")
    st.write("Prediksi:", "CVD" if pred == 1 else "No CVD")

    # =====================
    # SHAP (KernelExplainer)
    # =====================
    st.subheader("🧠 SHAP – Local Explanation")

    background = np.zeros((50, input_data.shape[1]))

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    shap_values = explainer.shap_values(
        input_data,
        nsamples=100
    )[1]  # class CVD

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        input_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)

    st.caption(
        "SHAP KernelExplainer digunakan karena AdaBoost "
        "tidak didukung oleh TreeExplainer."
    )

    # =====================
    # LIME (BACKGROUND PUTIH)
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba
    )

    lime_html = f"""
    <div style="
        background-color:white;
        padding:20px;
        border-radius:10px;
        box-shadow:0 0 10px rgba(0,0,0,0.3);
        overflow-x:auto;
    ">
    {exp.as_html()}
    </div>
    """

    st.components.v1.html(lime_html, height=850, scrolling=True)
