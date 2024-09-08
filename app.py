
import streamlit as st
import joblib

# Title of the app
st.title("AI-Generated Essay Detector")
st.write("Detect whether an essay is AI-generated or human-written.")

# Load the saved model and vectorizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('essay_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Text input for the essay
essay_input = st.text_area("Enter your essay here:", height=300)

# Button to analyze the essay
if st.button("Analyze"):
    if essay_input.strip() == "":
        st.warning("Please enter an essay for analysis.")
    else:
        new_essays = [essay_input]
        new_essays_tfidf = vectorizer.transform(new_essays)
        prediction = model.predict(new_essays_tfidf)[0]
        label = 'AI-Generated' if prediction == 1 else 'Human-Written'
        st.success(f"The essay is **{label}**.")
