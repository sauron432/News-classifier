import streamlit as st
import spacy
import joblib
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

model_file = "notebook/LR_classifier.pkl"
scaler_file = "notebook/scaler.pkl"
@st.cache_resource # This keeps the model in memory so it doesn't reload every click
def load_resources():
    try:
        with open(model_file,'rb') as file:
            model = joblib.load(file)
        with open(scaler_file,'rb') as file:
            scaler = joblib.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error(f'Error:{model_file} not found!')
        st.stop()

model, scaler = load_resources()
nlp = spacy.load('en_core_web_lg')
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

st.title("Fake News Classifier")
st.markdown("Enter a news headline or article below to check its authenticity.")

user_input = st.text_area("News Text", placeholder="Paste news content here...")

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 5. Prediction logic
        text = preprocess(user_input)
        input = nlp(text).vector.reshape(1,-1)
        prediction = model.predict(input)
        probability = model.predict_proba(input)
        label_map = {1: "Real News", 0: "Fake News"}
        result = label_map[prediction[0]]
        confidence = np.max(probability) * 100
        
        # 6. Display Results
        st.divider()
        st.subheader(f"Result: {result}")
        st.progress(confidence / 100)
        st.write(f"Confidence Score: {confidence:.2f}%")
        
        if prediction == 1:
            st.success("This news appears to be legitimate.")
        else:
            st.error("Warning! This news shows characteristics of being fake.")

# Footer
st.sidebar.info("This model uses spaCy Word Vectors and Logistic Regression.")