import streamlit as st

st.set_page_config(
    page_title="Prediksi Cardiovascular",
    layout="wide"
)

st.title("🫀 Sistem Prediksi Penyakit Cardiovascular")

st.markdown("""
Aplikasi ini menggunakan **AdaBoost Classifier** untuk memprediksi risiko
penyakit Cardiovascular Disease berdasarkan data klinis pasien.

### 📌 Fitur Aplikasi:
- Prediksi berbasis **file CSV**
- Prediksi **input manual**
- Interpretabilitas model (SHAP & LIME)
- Evaluasi korelasi fitur (Spearman & Kendall)

👉 Silakan pilih menu di sidebar.
""")

