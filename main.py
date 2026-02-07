import streamlit as st
import joblib

st.title("TEST ENV")

@st.cache_resource
def load_model():
    return joblib.load("src/adaboost_modelfix.pkl")

model = load_model()

st.success("✅ Model berhasil diload")
