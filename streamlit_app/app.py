import streamlit as st
import requests
from PIL import Image

st.title("Plant Seedling Classifier")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Predict"):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Predicted plant: {prediction}")
        else:
            st.error("Prediction failed. Check if the API is running.")