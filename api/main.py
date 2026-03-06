from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.predict import predict


app = FastAPI(title="Plant Seedling Classifier API")


@app.get("/")
def home():
    return {"message": "Plant classifier API is running"}


@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    prediction = predict(image)

    return {"prediction": prediction}