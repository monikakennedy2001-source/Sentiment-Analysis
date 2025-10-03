# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# ===============================
# Streamlit Page Setup
# ===============================
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="wide")
st.title("🎬 IMDB Movie Review Sentiment Analysis")

# ===============================
# Step 1: Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB-Dataset.csv")
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    df['label'] = df['sentiment'].map({'negative':0, 'positive':1})
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ===============================
# Step 2: Exploratory Data Analysis
# ===============================
st.subheader("Class Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Review Length Distribution")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.histplot(df['review_length'], bins=50, kde=True, color="purple", ax=ax2)
ax2.set_xlabel("Number of Words per Review")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

st.subheader("WordCloud - Positive Reviews")
pos_text = " ".join(df[df['label']==1]['review'])
pos_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.imshow(pos_wc, interpolation='bilinear')
ax3.axis("off")
st.pyplot(fig3)

st.subheader("WordCloud - Negative Reviews")
neg_text = " ".join(df[df['label']==0]['review'])
neg_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)
fig4, ax4 = plt.subplots(figsize=(10,5))
ax4.imshow(neg_wc, interpolation='bilinear')
ax4.axis("off")
st.pyplot(fig4)

# ===============================
# Step 3: Load Pre-trained Model & Tokenizer
# ===============================
model_path = os.path.join("models", "sentiment_lstm_final.h5")
tokenizer_path = os.path.join("models", "tokenizer.pkl")

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200  # same as used during training

# ===============================
# Step 4: Text Cleaning Function
# ===============================
def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Step 5: Prediction Function
# ===============================
def predict_review(review):
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(pad)[0][0]
    sentiment = "Positive 😀" if prob > 0.5 else "Negative 😞"
    return sentiment, prob

# ===============================
# Step 6: User Input for Prediction
# ===============================
st.subheader("Predict Sentiment for Your Review")
review_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        sentiment, prob = predict_review(review_input)
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Prediction Probability: {prob:.4f}")
