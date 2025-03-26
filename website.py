import torch
import streamlit as st
import plotly.express as px
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Model and Tokenizer
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion Labels (GoEmotions Dataset)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# Define emotions linked to depression
DEPRESSION_EMOTIONS = {"sadness", "disappointment", "grief", "remorse", "nervousness"}

# Emotion detection function
def detect_emotions(text):
    """Detect emotions and calculate depression probability."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    emotion_scores = {EMOTION_LABELS[i]: scores[i] for i in range(len(EMOTION_LABELS))}

    top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate depression probability
    depression_score = sum(scores[EMOTION_LABELS.index(e)] for e in DEPRESSION_EMOTIONS if e in EMOTION_LABELS)
    depression_chance = min(depression_score * 100, 100)

    return emotion_scores, top_emotions, depression_chance

# Streamlit UI
st.set_page_config(page_title="Emotion & Depression Analysis", layout="wide", page_icon="😃")

st.title("🎭 Emotion & Depression Analysis")
st.markdown("##### AI-powered emotion detection with depression risk analysis.")

# User Input
user_text = st.text_area("💬 Enter a sentence:", "")

if st.button("🔍 Analyze Emotion"):
    if user_text.strip():
        with st.spinner("Analyzing emotions... 🚀"):
            emotion_scores, top_emotions, depression_chance = detect_emotions(user_text)

            # Display detected emotions
            st.subheader("🎭 Detected Emotions")
            for emotion, confidence in top_emotions:
                st.write(f"**{emotion.capitalize()}** - Confidence: `{confidence:.2f}`")

            # Depression Analysis
            st.subheader("🧠 Depression Probability")
            st.write(f"**Chances of Depression: `{depression_chance:.2f}%`**")

            # Plot: Emotion Distribution (Bar Chart)
            fig_bar = px.bar(
                x=list(emotion_scores.keys()), y=list(emotion_scores.values()),
                labels={'x': "Emotions", 'y': "Confidence Score"},
                title="Emotion Confidence Scores",
                color=list(emotion_scores.keys()),
                template="plotly_dark"
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Plot: Depression Risk (Gauge Meter)
            fig_gauge = px.bar_polar(
                r=[depression_chance], theta=["Depression Risk"],
                title="Depression Risk Meter",
                color_discrete_sequence=["red"] if depression_chance > 50 else ["green"]
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Pie Chart: Top 3 Emotions
            top_labels, top_values = zip(*top_emotions)
            fig_pie = px.pie(
                names=top_labels, values=top_values,
                title="Top Detected Emotions",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("⚠ Please enter a sentence to analyze.")
