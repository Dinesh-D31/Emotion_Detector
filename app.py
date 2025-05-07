# Imports
import streamlit as st
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Page Configuration
st.set_page_config(
    page_title="Emotion Detector 🧠",
    page_icon="🧠",
    layout="centered"
)

# Load Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Model, Vectorizer and Label Binarizer
model = joblib.load('C:/Users/Asus/Documents/projects/emotion_classifier_model.pkl')
vectorizer = joblib.load("C:/Users/Asus/Documents/projects/tfidf_vectorizer.pkl")
mlb = joblib.load('C:/Users/Asus/Documents/projects/mlb.pkl') 

# Title
st.markdown("""
    <h1 style='text-align: center; color: #6C63FF;'>Emotion Detector 🧠</h1>
    <p style='text-align: center; font-size:18px;'>Feel free to share your emotions 💬🧠😊😠</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Text Input
user_input = st.text_area("Enter your text here...", height=150)

# Predict Button
if st.button("Predict Emotion"):
    if user_input:
        # Vectorize input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)

        # Decode predictions
        predicted_labels = mlb.inverse_transform(prediction)

        # Show result
        if predicted_labels and predicted_labels[0]:
            st.success(f"🎯 Predicted Emotions: {', '.join(predicted_labels[0])}")
        else:
            st.warning("⚠️ No emotions detected.")
    else:
        st.warning("Please enter some text for prediction.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made with ❤️ by Dinesh</p>
""", unsafe_allow_html=True)

st.markdown("© 2025 Dinesh. All rights reserved.", unsafe_allow_html=True)



