import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Prediksi Cardiovascular – Global Input",
    layout="wide"
)

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    return joblib.load("adaboost_modelfix.pkl")

model = load_model()
feature_names = model.feature_names_in_

st.title("📂 Prediksi Penyakit Cardiovascular Berbasis Explainable AI")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# =====================
# MAIN LOGIC
# =====================
if uploaded_file:

    # =====================
    # LOAD DATA
    # =====================
    df = pd.read_csv(uploaded_file)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # =====================
    # PREPROCESSING
    # =====================
    for col in ["sex", "exang", "fbs"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    mapping_binary = {
        "sex": {"male": 1, "female": 0},
        "exang": {"yes": 1, "no": 0},
        "fbs": {"true": 1, "false": 0}
    }

    for col, mp in mapping_binary.items():
        if col in df.columns:
            df[col] = df[col].map(mp)

    X_raw = df.drop(columns=["num", "id"], errors="ignore")

    # =====================
    # FEATURE MAPPING
    # =====================
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

    X_model = X_raw.rename(columns=df_to_model_mapping)

    for col in feature_names:
        if col not in X_model.columns:
            X_model[col] = 0

    X_model = (
        X_model[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # =====================
    # PREDICTION
    # =====================
    df["prediction"] = model.predict(X_model)
    df["probability"] = model.predict_proba(X_model)[:, 1]

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================
    # SHAP GLOBAL (TREE EXPLAINER – AMAN)
    # =====================
    st.subheader("🧠 SHAP – Global Feature Importance")

    import shap

    explainer = shap.TreeExplainer(model)

    sample_X = X_model.sample(
        min(100, len(X_model)),
        random_state=42
    )

    shap_values = explainer.shap_values(sample_X)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        sample_X,
        show=False
    )
    st.pyplot(fig)

    st.caption(
        "SHAP TreeExplainer digunakan karena efisien dan stabil "
        "untuk model berbasis tree seperti AdaBoost."
    )

    # =====================
    # LIME LOCAL (SAFE MODE)
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    import lime.lime_tabular

    idx = st.slider(
        "Pilih indeks data",
        0,
        len(X_model) - 1,
        0
    )

    lime_background = X_model.sample(
        min(50, len(X_model)),
        random_state=42
    ).values

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        X_model.iloc[idx].values,
        model.predict_proba,
        num_features=5
    )

    fig_lime = exp.as_pyplot_figure(label=1)
    fig_lime.patch.set_facecolor("white")
    st.pyplot(fig_lime)

    # =====================
    # KESIMPULAN
    # =====================
    st.subheader("📌 Kesimpulan")

    st.markdown("""
    - **SHAP digunakan untuk analisis global**, menilai kontribusi fitur
      terhadap seluruh dataset.
    - **LIME digunakan untuk analisis lokal**, menjelaskan keputusan model
      pada satu pasien tertentu.
    - Pendekatan ini menjaga **keseimbangan interpretabilitas dan performa sistem**.
    """)

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")

