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
    age = st.number_input("Usia", 1, 120, 45)

    sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    sex = 1 if sex == "Laki-laki" else 0

    dataset = st.selectbox("Dataset Asal", [0, 1, 2, 3])  
    # contoh: 0=Cleveland, 1=Hungary, dst (harus sesuai encoding kamu)

    cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])

    trestbps = st.number_input("Tekanan Darah", 50, 250, 130)
    chol = st.number_input("Kolesterol", 80, 600, 230)

    fbs = st.selectbox("Gula Darah Puasa >120 mg/dl", [0, 1])

    restecg = st.selectbox("Hasil ECG (restecg)", [0, 1, 2])

    thalch = st.number_input("Denyut Jantung Maksimum (thalch)", 50, 250, 150)

    exang = st.selectbox("Angina akibat olahraga (exang)", [0, 1])

    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)

    slope = st.selectbox("Slope ST", [0, 1, 2])

    ca = st.selectbox("Jumlah Pembuluh Darah Mayor (ca)", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])


    submitted = st.form_submit_button("Prediksi")

# Convert input to numerical
sex_value = 1 if sex == "Laki-laki" else 0

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_data = np.array([[
        age, sex, dataset, cp, trestbps, chol, fbs,
        restecg, thalch, exang, oldpeak, slope, ca, thal
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

