import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import matplotlib.pyplot as plt
import json

# =====================================================
# HELPER: EXTRACT PROBABILITY FROM ONNX OUTPUT
# =====================================================
def extract_proba(outputs):
    """
    Convert ONNX outputs to (N, 2) probability matrix
    Safe for AdaBoost / Tree-based ONNX models
    """
    proba_raw = outputs[1]

    # Case 1: list of dicts [{0:p0,1:p1}, ...]
    if isinstance(proba_raw[0], dict):
        return np.array([[p[0], p[1]] for p in proba_raw])

    # Case 2: 1D array (only class-1 prob)
    if np.ndim(proba_raw) == 1:
        return np.column_stack([1 - proba_raw, proba_raw])

    # Case 3: already (N,2)
    return np.array(proba_raw)


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
    # ALIGN FEATURES WITH MODEL
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
    # PREDICTION (ONNX - FIXED)
    # =====================
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: X_model.values}
    )

    preds = outputs[0]
    proba_matrix = extract_proba(outputs)
    probs = proba_matrix[:, 1]

    df["prediction"] = preds
    df["probability"] = probs

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================
    # INTERPRETATION
    # =====================
    st.subheader("📝 Interpretasi Hasil Prediksi")

    total = len(df)
    total_cvd = int(df["prediction"].sum())

    st.markdown(f"""
    Dari **{total} data pasien** yang dianalisis:

    - **{total_cvd} pasien** diprediksi **mengalami CVD**
    - **{total - total_cvd} pasien** diprediksi **tidak mengalami CVD**
    """)

    # =====================================================
    # LIME LOCAL EXPLANATION (ONNX SAFE)
    # =====================================================
    st.subheader("🧩 LIME – Local Explanation")

    idx = st.slider("Pilih indeks data", 0, len(X_model) - 1, 0)

    def onnx_predict_proba(x):
        outputs = session.run(
            OUTPUT_NAMES,
            {INPUT_NAME: x.astype(np.float32)}
        )
        return extract_proba(outputs)

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

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
