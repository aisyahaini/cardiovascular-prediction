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

st.title("📂 Prediksi Penyakit Cardiovascular dari File CSV")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    # =====================
    # LOAD & CLEAN DATA
    # =====================
    df = pd.read_csv(uploaded_file)

    # strip spasi di seluruh data
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # =====================
    # NORMALISASI STRING
    # =====================
    for col in ["sex", "exang", "fbs"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # =====================
    # ENCODING KATEGORIKAL
    # =====================
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 1, "female": 0})

    if "exang" in df.columns:
        df["exang"] = df["exang"].map({"yes": 1, "no": 0})

    if "fbs" in df.columns:
        df["fbs"] = df["fbs"].map({"true": 1, "false": 0})

    # =====================
    # DROP TARGET & ID
    # =====================
    X_raw = df.drop(columns=["num", "id"], errors="ignore")

    # =====================
    # MAPPING DATASET → MODEL
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

    X_model = X_model[model.feature_names_in_]

    # pastikan numerik & bebas NaN
    X_model = (
        X_model
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # =====================
    # PREDIKSI
    # =====================
    preds = model.predict(X_model)
    probs = model.predict_proba(X_model)[:, 1]

    df["prediction"] = preds
    df["probability"] = probs

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================
    # SHAP GLOBAL 
    # =====================
    st.subheader("🧠 SHAP – Global Feature Importance")

    # ambil background kecil (biar cepat)
    background = shap.sample(X_model, 50, random_state=42)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    # hitung SHAP (kelas 1 = CVD)
    shap_values = explainer.shap_values(
        X_model.iloc[:100],  # batasi biar cepat
        nsamples=100
    )

    shap_values = shap_values[1]  # kelas positif

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_model.iloc[:100],
        feature_names=X_model.columns,
        show=False
    )
    st.pyplot(fig)

    st.markdown("""
    **Kernel SHAP** digunakan karena model AdaBoost
    tidak didukung secara langsung oleh TreeExplainer.
    Metode ini bersifat model-agnostic dan valid
    untuk semua jenis classifier.
    """)



    # =====================
    # LIME (LOCAL EXPLANATION)
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

    # === FIX BACKGROUND PUTIH ===
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

    st.components.v1.html(lime_html, height=650, scrolling=True)

# =====================
# KORELASI FITUR (SPEARMAN & KENDALL)
# =====================
st.subheader("📈 Korelasi Fitur terhadap Target (num)")

if "num" in df.columns:
    y = df["num"]

    spearman_results = {}
    kendall_results = {}

    for col in X_raw.columns:
        # pastikan numerik & tidak konstan
        x = pd.to_numeric(df[col], errors="coerce")

        if x.nunique() > 1:
            spearman_results[col] = x.corr(y, method="spearman")
            kendall_results[col] = x.corr(y, method="kendall")
        else:
            spearman_results[col] = np.nan
            kendall_results[col] = np.nan

    corr_df = pd.DataFrame({
        "Spearman": spearman_results,
        "Kendall": kendall_results
    })

    # absolut → cari pengaruh terbesar
    corr_df["Abs_Spearman"] = corr_df["Spearman"].abs()
    corr_df["Abs_Kendall"] = corr_df["Kendall"].abs()

    corr_df = corr_df.sort_values(
        by="Abs_Spearman",
        ascending=False
    )

    st.markdown("### 🔝 Top 10 Fitur dengan Korelasi Terkuat")
    st.dataframe(
        corr_df[["Spearman", "Kendall"]].head(10)
    )

    # =====================
    # VISUALISASI
    # =====================
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_df.head(10)[["Spearman", "Kendall"]].plot.bar(ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Feature Correlation")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Kolom target 'num' tidak ditemukan, korelasi tidak dapat dihitung.")
