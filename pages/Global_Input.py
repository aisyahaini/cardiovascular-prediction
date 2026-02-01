import streamlit as st
import pandas as pd
import numpy as np
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

#st.set_page_config(page_title="Prediksi Cardiovascular", layout="wide")
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

    # =====================
    # FEATURE SELECTION
    # =====================
    X_raw = df.drop(columns=["num", "id"], errors="ignore")

    # =====================
    # FEATURE NAME MAPPING
    # =====================
    shap_to_df_mapping = {
        "num_major_vessels": "ca",
        "chest_pain_type": "cp",
        "thalassemia_type": "thal",
        "st_slope_type": "slope",
        "cholesterol": "chol",
        "st_depression": "oldpeak",
        "resting_blood_pressure": "trestbps",
        "sex": "sex",
        "age": "age",
        "max_heart_rate_achieved": "thalch",
        "Restecg": "restecg",
        "exercise_induced_angina": "exang",
        "fasting_blood_sugar": "fbs",
        "country": "dataset"
    }

    df_to_model_mapping = {v: k for k, v in shap_to_df_mapping.items()}
    X_model = X_raw.rename(columns=df_to_model_mapping)

    # =====================
    # SAMAKAN FITUR MODEL
    # =====================
    for col in model.feature_names_in_:
        if col not in X_model.columns:
            X_model[col] = 0

    X_model = (
        X_model[model.feature_names_in_]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # =====================
    # PREDICTION
    # =====================
    preds = model.predict(X_model)
    probs = model.predict_proba(X_model)[:, 1]

    df["prediction"] = preds
    df["probability"] = probs

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================
    # INTERPRETASI HASIL
    # =====================
    st.subheader("📝 Interpretasi Hasil Prediksi")

    total = len(df)
    total_cvd = df["prediction"].sum()

    st.markdown(f"""
    Dari **{total} data pasien** yang dianalisis:

    - **{total_cvd} pasien** diprediksi **mengalami penyakit kardiovaskular (CVD)**
    - **{total - total_cvd} pasien** diprediksi **tidak mengalami CVD**

    Nilai probabilitas menunjukkan tingkat keyakinan model terhadap prediksi.
    """)

    # =====================
    # SHAP GLOBAL
    # =====================
    st.subheader("🧠 SHAP – Global Feature Importance")

    background = shap.sample(X_model, min(50, len(X_model)), random_state=42)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    shap_values = explainer.shap_values(
        X_model.iloc[:min(100, len(X_model))],
        nsamples=100
    )[1]

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_model.iloc[:min(100, len(X_model))],
        feature_names=X_model.columns,
        show=False
    )
    st.pyplot(fig)

    st.markdown("""
    SHAP memberikan **penjelasan global** dengan menghitung kontribusi setiap fitur
    terhadap seluruh dataset.
    """)

    # =====================
    # LIME LOCAL
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    idx = st.slider(
        "Pilih indeks data",
        0,
        len(X_model) - 1,
        0
    )

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_model.values,
        feature_names=X_model.columns,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        X_model.iloc[idx].values,
        model.predict_proba
    )

    # ===== FORCE WHITE BACKGROUND =====
    lime_html = f"""
    <div style="
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 100%;
        overflow-x: auto;
    ">

    <style>
    /* Paksa semua elemen LIME jadi putih */
    body {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    table {{
        background-color: #ffffff !important;
    }}

    th, td {{
        color: #000000 !important;
    }}

    svg {{
        background-color: #ffffff !important;
    }}
    </style>

    {exp.as_html()}
    </div>
    """

    st.components.v1.html(
        lime_html,
        height=650,
        scrolling=True
    )
    
    # =====================
    # KORELASI FITUR
    # =====================
    st.subheader("📈 Korelasi Fitur terhadap Target")

    if "num" in df.columns:
        y = df["num"]

        corr_data = []
        for col in X_raw.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            if x.nunique() > 1:
                corr_data.append([
                    col,
                    x.corr(y, method="spearman"),
                    x.corr(y, method="kendall")
                ])

        corr_df = pd.DataFrame(
            corr_data,
            columns=["Feature", "Spearman", "Kendall"]
        ).set_index("Feature")

        corr_df["Abs_Spearman"] = corr_df["Spearman"].abs()
        corr_df = corr_df.sort_values("Abs_Spearman", ascending=False)

        st.markdown("### 🔝 Top 10 Fitur dengan Korelasi Terkuat")
        st.dataframe(corr_df[["Spearman", "Kendall"]].head(10))

        fig, ax = plt.subplots()
        corr_df.head(10)[["Spearman", "Kendall"]].plot.bar(ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # =====================
        # KESIMPULAN
        # =====================
        top5 = corr_df.head(5).index.tolist()

        st.subheader("📌 Kesimpulan Akhir")
        st.markdown(f"""
        Berdasarkan analisis:

        **5 fitur teratas yang paling berkaitan dengan penyakit kardiovaskular**
        (berdasarkan Spearman & Kendall):

        1. {top5[0]}
        2. {top5[1]}
        3. {top5[2]}
        4. {top5[3]}
        5. {top5[4]}

        **Kesimpulan:**
        - Model AdaBoost mampu memprediksi risiko CVD dengan baik.
        - **SHAP lebih unggul untuk analisis global** karena mengevaluasi seluruh data.
        - **LIME lebih tepat untuk analisis individual** atau satu pasien.
        """)

    else:
        st.warning("Kolom target 'num' tidak tersedia.")

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
