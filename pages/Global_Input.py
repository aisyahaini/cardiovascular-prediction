import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import matplotlib.pyplot as plt
import json

# =====================================================
# LOAD ONNX MODEL & FEATURE NAMES
# =====================================================
@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(
        "adaboost_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    return session, feature_names, input_name, output_names


session, FEATURE_NAMES, INPUT_NAME, OUTPUT_NAMES = load_onnx_model()

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("📂 Prediksi Penyakit Cardiovascular (ONNX + Explainable AI)")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# =====================================================
# MAIN LOGIC
# =====================================================
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
    for col in FEATURE_NAMES:
        if col not in X_model.columns:
            X_model[col] = 0

    X_model = (
        X_model[FEATURE_NAMES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.float32)
    )

    # =====================
    # PREDICTION (ONNX)
    # =====================
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: X_model.values}
    )

    preds = outputs[0]
    probs = outputs[1][:, 1]

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

    # =====================================================
    # LIME LOCAL EXPLANATION (ONNX WRAPPER)
    # =====================================================
    st.subheader("🧩 LIME – Local Explanation")

    idx = st.slider("Pilih indeks data", 0, len(X_model) - 1, 0)

    def onnx_predict_proba(x):
        result = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: x.astype(np.float32)}
        )
        return result[1]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_model.values,
        feature_names=FEATURE_NAMES,
        class_names=["No CVD", "CVD"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        X_model.iloc[idx].values,
        onnx_predict_proba
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
