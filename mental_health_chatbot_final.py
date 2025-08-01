import streamlit as st
import pandas as pd
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import together
import os

# Load and clean data
df1 = pd.read_csv("balanced_sentiment_data.csv")
df2 = pd.read_csv("combined_emotion.csv")
df3 = pd.read_csv("combined_sentiment_data.csv")

df1 = df1.rename(columns={'sentence': 'text', 'sentiment': 'label'})
df2 = df2.rename(columns={'sentence': 'text', 'emotion': 'label'})
df3 = df3.rename(columns={'sentence': 'text', 'sentiment': 'label'})

df = pd.concat([df1, df2, df3], ignore_index=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Emotion Detection Model (Hugging Face)
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
emotion_labels = ['anger', 'joy', 'optimism', 'sadness']

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    top_idx = probs.argmax()
    return emotion_labels[top_idx], float(probs[top_idx])

# Together AI setup
api_key = os.getenv("TOGETHER_API_KEY")
together.api_key = api_key

def generate_response_together(user_input, emotion):
    prompt = f"""
You are a compassionate mental health journaling assistant.

User is feeling: {emotion}
Their journal entry: "{user_input}"

Please respond with:
- A warm and empathetic message
- A quote from Gita, Quran, Bible, or a motivational speaker
- One calming or uplifting suggestion
"""
    response = together.Complete.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    return response["choices"][0]["text"].strip()

# Streamlit UI
st.set_page_config(page_title="Mental Health Journaling Chatbot", layout="centered")
st.title("💬 Mental Health Journaling Companion")
st.markdown("Write your thoughts below and receive compassionate support, quotes, and suggestions.")

user_input = st.text_area("📝 Your Journal Entry", height=150)

if st.button("🧠 Analyze & Respond"):
    if user_input.strip():
        emotion, prob = detect_emotion(user_input)
        response = generate_response_together(user_input, emotion)

        st.markdown(f"### 🎯 Detected Emotion: `{emotion}` ({prob:.2f})")
        st.markdown("### 🤖 Chatbot Response")
        st.success(response)

        if st.checkbox("💾 Save this entry"):
            log = pd.DataFrame([[user_input, emotion, response]], columns=["Journal", "Emotion", "Response"])
            log.to_csv("journal_log.csv", mode='a', header=False, index=False)
            st.toast("Saved to journal_log.csv")
    else:
        st.warning("Please write something before submitting.")
