import joblib 
import spacy


def predict(text):
    try:
        with open('model/LR_classifier.pkl','rb') as file:
            model = joblib.load(file)
        print("Model loaded successfully!")
        nlp = spacy.load("en_core_web_lg")
        input = nlp(text).vector.reshape(1,-1)
        prediction = model.predict(input)
        probability = model.predict_proba(input)
        label_map = {1: "Real News", 0: "Fake News"}
        result = label_map[prediction[0]]
        print(f"Prediction: {result}")
        print(f"Confidence: {max(probability[0]) * 100:.2f}%")
    except Exception as e:
        print("Error: ",e)