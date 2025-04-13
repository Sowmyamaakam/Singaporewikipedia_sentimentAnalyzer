import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the vectorizer and model
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Class labels
class_labels = model.classes_

# App title
st.title("Sentiment Analysis App 🤖")
st.subheader("Predict the sentiment of your text using Logistic Regression")

# User input
text = st.text_area("Enter your text here")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Vectorize the input
        X_input = vectorizer.transform([text])
        
        # Prediction and probabilities
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        # Map 0/1 to Negative/Positive
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        # Show prediction
        st.success(f"🎯 **Prediction:** {sentiment_label}")
        st.info(f"🔍 **Model Used:** Logistic Regression")

        # Bar chart for class probabilities
        st.subheader("📊 Class Probabilities")
        fig, ax = plt.subplots()
        sns.barplot(x=class_labels, y=probabilities, ax=ax, palette="viridis")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Sentiment Class")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # WordCloud
        st.subheader("☁️ Word Cloud of Your Input")
        wordcloud = WordCloud(width=800, height=300, background_color="white").generate(text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
