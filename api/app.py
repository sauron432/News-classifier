from fastapi import FastAPI, Body
import os
import joblib
import spacy

from src.data_preprocess import preprocess

app = FastAPI(title="News classification")


nlp = spacy.load('en_core_web_lg')
MODEL_PATH = "model/LR_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found!")

with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

@app.get("/")
def test():
    return {"status":"Running!"}

@app.post("/identify")
def classify(text: str = Body(..., media_type='text/plain')):
    try:
        preprocess_input = preprocess(text)
        input = nlp(preprocess_input).vector.reshape(1,-1)
        prediction = model.predict(input)
        probability = model.predict_proba(input)
        label_map = {1: "Real News", 0: "Fake News"}
        result = label_map[prediction[0]]
        return {
            "prediction":result,
            "confidence":max(probability[0]) * 100
        }
        
    except Exception as e:
            return {"error": str(e)}
