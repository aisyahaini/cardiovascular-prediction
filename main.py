import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("adaboost_modelfix.pkl")  # ganti sesuai nama file kamu
    return model

model = load_model()

st.title("🫀 Prediksi Penyakit Cardiovascular")
st.write("Masukkan data pasien untuk memprediksi risiko penyakit jantung.")

# ===============================
# INPUT FORM
# ===============================
with st.form("input_form"):
    age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
    trestbps = st.number_input("Tekanan Darah (resting blood pressure)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Kolesterol (mg/dL)", min_value=80, max_value=600, value=230)
    thalach = st.number_input("Denyut Jantung Maksimum (thalach)", min_value=50, max_value=250, value=150)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
    exang = st.selectbox("Angina Induced by Exercise", [0, 1])
    slope = st.selectbox("Slope ST", [0, 1, 2])

    submitted = st.form_submit_button("Prediksi")

# Convert input to numerical
sex_value = 1 if sex == "Laki-laki" else 0

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_data = np.array([[
        age, trestbps, chol, thalach, oldpeak, cp,
        sex_value, fbs, exang, slope
    ]])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # probabilitas CVD

    st.subheader("🔍 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Pasien diprediksi **Berisiko Cardiovascular Disease**")
        st.write(f"Probabilitas risiko: **{proba:.2f}**")
    else:
        st.success(f"✔️ Pasien diprediksi **Tidak Berisiko Cardiovascular Disease**")
        st.write(f"Probabilitas risiko: **{proba:.2f}**")

    # ===============================
    # KESIMPULAN OTOMATIS
    # ===============================
    st.subheader("📝 Kesimpulan")
    
    if prediction == 1:
        st.write(
            """
            Berdasarkan data yang diinputkan, model memprediksi bahwa pasien 
            memiliki **risiko terkena penyakit cardiovascular**.  
            Disarankan untuk melakukan pemeriksaan lanjutan seperti:
            - pemeriksaan EKG,
            - tes treadmill,
            - konsultasi dokter spesialis jantung.

            Perubahan gaya hidup seperti aktivitas fisik teratur, 
            mengurangi konsumsi garam & lemak, serta monitoring tekanan darah 
            juga dapat membantu menurunkan risiko.
            """
        )
    else:
        st.write(
            """
            Berdasarkan data yang diinputkan, model memprediksi bahwa pasien 
            **tidak memiliki risiko signifikan terhadap penyakit cardiovascular**.  
            Namun, tetap dianjurkan menjaga gaya hidup sehat dengan:
            - olahraga rutin,
            - menjaga berat badan,
            - mengelola stres,
            - memonitor tekanan darah & kolesterol secara berkala.
            """
        )

    st.info("Hasil prediksi ini bukan diagnosis medis dan tetap memerlukan pemeriksaan dokter.")

