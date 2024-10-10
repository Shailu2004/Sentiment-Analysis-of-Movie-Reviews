import streamlit as st
import pickle

# Load the trained  model and TF-IDF vectorizer
with open('sentiment_model_lr.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Streamlit app title and description
st.title("Sentiment Analysis of Movie Reviews")
st.write("Enter a movie review and find out if the sentiment is Negative, Neutral, or Positive.")

# Text input for the review
user_input = st.text_area("Enter your review here:")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a review before clicking the Predict button.")
    else:
        # Convert the user input into TF-IDF features
        input_features = tfidf.transform([user_input])
        
        # Predict the sentiment
        prediction = model.predict(input_features)[0]

        # Map the prediction to the sentiment label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"The sentiment of the review is: {sentiment_map[prediction]}")
