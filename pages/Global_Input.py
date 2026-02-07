import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    return joblib.load("adaboost_modelfix.pkl")

model = load_model()

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
    # INTERPRETASI
    # =====================
    st.subheader("📝 Interpretasi Hasil Prediksi")

    total = len(df)
    total_cvd = int(df["prediction"].sum())

    st.markdown(f"""
    Dari **{total} data pasien** yang dianalisis:

    - **{total_cvd} pasien** diprediksi **mengalami penyakit kardiovaskular (CVD)**
    - **{total - total_cvd} pasien** diprediksi **tidak mengalami CVD**
    """)

    # =====================
    # GLOBAL FEATURE IMPORTANCE
    # =====================
    st.subheader("🧠 Global Feature Importance (Permutation Importance)")

    y_for_importance = df["prediction"]

    result = permutation_importance(
        model,
        X_model,
        y_for_importance,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X_model.columns,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance_df.head(10))

    fig, ax = plt.subplots()
    importance_df.head(10).plot.barh(
        x="Feature",
        y="Importance",
        ax=ax,
        legend=False
    )
    ax.invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

    # =====================
    # LIME LOCAL EXPLANATION
    # =====================
    st.subheader("🧩 LIME – Local Explanation")

    idx = st.slider("Pilih indeks data", 0, len(X_model) - 1, 0)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_model.values,
        feature_names=X_model.columns.tolist(),
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        X_model.iloc[idx].values,
        model.predict_proba
    )

    st.components.v1.html(exp.as_html(), height=650, scrolling=True)

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

        st.dataframe(corr_df[["Spearman", "Kendall"]].head(10))

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
