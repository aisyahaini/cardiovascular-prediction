import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import lime.lime_tabular
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# =====================================================
# HELPER: EXTRACT PROBABILITY FROM ONNX OUTPUT
# =====================================================
def extract_proba(outputs):
    proba_raw = outputs[1]

    if isinstance(proba_raw[0], dict):
        return np.array([[p[0], p[1]] for p in proba_raw])

    if np.ndim(proba_raw) == 1:
        return np.column_stack([1 - proba_raw, proba_raw])

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
st.title("📂 Prediksi Penyakit Cardiovascular (ONNX + XAI)")

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
    # PREDICTION
    # =====================
    outputs = session.run(
        OUTPUT_NAMES,
        {INPUT_NAME: X_model.values}
    )

    preds = outputs[0]
    proba_matrix = extract_proba(outputs)

    df["prediction"] = preds
    df["probability"] = proba_matrix[:, 1]

    st.subheader("📌 Hasil Prediksi")
    st.dataframe(df.head())

    # =====================
    # INTERPRETATION
    # =====================
    st.subheader("📝 Interpretasi")

    total = len(df)
    total_cvd = int(df["prediction"].sum())

    st.markdown(f"""
    - Total pasien: **{total}**
    - Prediksi CVD: **{total_cvd}**
    - Tidak CVD: **{total - total_cvd}**
    """)

    # =====================================================
    # LIME EXPLANATION
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
        onnx_predict_proba,
        num_features=len(FEATURE_NAMES)
    )

    st.components.v1.html(exp.as_html(), height=650, scrolling=True)

    # =====================================================
    # SHAP-LIKE GLOBAL EXPLANATION (NO SHAP LIB)
    # =====================================================
    st.subheader("📊 Global Feature Importance (SHAP-like)")

    sample_size = min(50, len(X_model))
    sampled_idx = np.random.choice(len(X_model), sample_size, replace=False)

    feature_importance = defaultdict(list)

    for i in sampled_idx:
        explanation = lime_explainer.explain_instance(
            X_model.iloc[i].values,
            onnx_predict_proba,
            num_features=len(FEATURE_NAMES)
        )

        for feature, weight in explanation.as_list():
            fname = feature.split(" ")[0]
            feature_importance[fname].append(abs(weight))

    shap_like_importance = {
        f: np.mean(w) for f, w in feature_importance.items()
    }

    shap_df = (
        pd.DataFrame({
            "Feature": shap_like_importance.keys(),
            "Importance": shap_like_importance.values()
        })
        .sort_values(by="Importance", ascending=False)
    )

    # =====================
    # PLOT
    # =====================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        shap_df["Feature"],
        shap_df["Importance"]
    )
    ax.set_xlabel("Mean |Contribution|")
    ax.set_title("Global Feature Importance (SHAP-like)")
    ax.invert_yaxis()

    st.pyplot(fig)

else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
