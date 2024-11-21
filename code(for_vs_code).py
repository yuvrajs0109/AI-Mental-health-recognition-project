import os
import zipfile
import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Tokenize and clean text
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load pre-trained emotion detection model
emotion_model = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base', device=0)

# Define function to predict emotion
def predict_emotion(text):
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Get the emotion prediction using the pre-trained model
    emotion = emotion_model(cleaned_text)[0]['label']
    return emotion

# Define Streamlit application interface
def main():
    st.title("Mental Health Emotion Detection")
    st.subheader("Enter text to predict the emotion:")

    # Input text box
    user_input = st.text_area("Enter your text here:")

    if st.button("Predict Emotion"):
        if user_input:
            # Get the predicted emotion
            emotion = predict_emotion(user_input)
            st.write(f"Predicted Emotion: {emotion}")
        else:
            st.write("Please enter some text.")

# Run the app
if __name__ == "__main__":
    main()

