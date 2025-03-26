import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random
import requests

# Streamlit Page Config
st.set_page_config(page_title="Emotion & Depression Analysis", layout="wide")

# Load Lottie Animation
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_kt8n7oeq.json"
lottie_animation = load_lottie(lottie_url)

# Custom CSS for Styling
st.markdown("""
    <style>
        .metric-box {
            background: #282c34;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(255, 215, 0, 0.3);
            text-align: center;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Animation
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>Emotion & Depression Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Answer the questions below to analyze emotions & depression risk</h4>", unsafe_allow_html=True)

st.markdown("---")

# Questions for Emotion & Depression Detection
questions = [
    "How often do you feel happy or joyful?", "Do you feel sad or down frequently?", "How often do you feel anxious or nervous?",
    "Do you experience sudden mood swings?", "Do you feel lonely or isolated?", "How often do you have trouble sleeping?",
    "Do you feel overwhelmed by daily tasks?", "How often do you feel hopeless about the future?", "Do you find it hard to enjoy things you used to love?",
    "How often do you feel fatigued or lack energy?", "Do you feel irritable or easily frustrated?", "Do you have difficulty concentrating or making decisions?",
    "How often do you feel worthless or guilty?", "Do you feel like withdrawing from social activities?", "How often do you experience physical symptoms like headaches or body pain without a clear cause?",
]

# Arrange questions in 5 rows x 3 columns
total_questions = len(questions)
responses = []
cols = st.columns(3)

for i in range(5):  # 5 Rows
    for j in range(3):  # 3 Columns
        idx = i * 3 + j
        if idx < total_questions:
            with cols[j]:
                response = st.slider(questions[idx], min_value=0, max_value=10, value=5, format="%d")
                responses.append(response)

# Submit Button
if st.button("Analyze Results"):
    st.markdown("---")

    # Emotion & Depression Analysis Logic
    emotion_scores = {
        "Joy": round(responses[0] / 10, 2),
        "Sadness": round(responses[1] / 10, 2),
        "Anxiety": round(responses[2] / 10, 2),
        "Hopelessness": round(responses[7] / 10, 2),
        "Fatigue": round(responses[9] / 10, 2),
        "Social Withdrawal": round(responses[13] / 10, 2),
    }

    # Depression Probability Calculation (Improved)
    weights = {
        "Sadness": 0.2,
        "Anxiety": 0.15,
        "Hopelessness": 0.25,
        "Fatigue": 0.2,
        "Social Withdrawal": 0.2,
    }

    weighted_depression_score = sum(weights[key] * emotion_scores[key] for key in weights)
    depression_risk = min(1.0, max(0.0, weighted_depression_score * (1 - 0.5 * emotion_scores["Joy"])))
    depression_risk_percentage = round(depression_risk * 100, 2)

    df = pd.DataFrame(emotion_scores.items(), columns=['Emotion', 'Score']).sort_values(by="Score", ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Emotion Scores")
        fig_bar = px.bar(df, x="Emotion", y="Score", color="Emotion", title="Emotion Analysis", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.subheader("🎭 Emotion Distribution")
        fig_pie = px.pie(df, names="Emotion", values="Score", title="Emotion Share", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Depression Probability
    st.markdown("<h2 style='text-align: center; color: #FFD700;'>📈 Depression Probability</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(f"""
            <div class='metric-box'>
                <h2>🧠 {depression_risk_percentage:.2f}%</h2>
                <p>Estimated Depression Risk</p>
            </div>
        """, unsafe_allow_html=True)

    # Weekly Depression Trends
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    risk_values = [random.uniform(0.2, 0.9) for _ in range(7)]
    fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True)
    fig_line.update_layout(xaxis_title="Day", yaxis_title="Depression Risk Level")
    st.plotly_chart(fig_line, use_container_width=True)

    # Personalized Advice Based on Risk Level
    if depression_risk_percentage > 70:
        advice = "🚨 **High risk detected!** Consider seeking professional help. You're not alone. 💙"
    elif depression_risk_percentage > 40:
        advice = "⚠️ **Moderate risk.** Try self-care, meditation, and reaching out to friends or family."
    else:
        advice = "😊 **Low risk!** Keep up your healthy habits and stay mindful of your emotions."

    st.warning(advice)

    # Detailed Data Table
    st.subheader("📜 Detailed Emotion & Depression Report")
    st.dataframe(df)

st.markdown("<h3 style='text-align: center; color: #FFD700;'>Stay Strong & Take Care! 😊</h3>", unsafe_allow_html=True)
st.json(lottie_animation)
st.markdown("<h6 style='text-align: center; color: grey;'>Built with ❤️ using Streamlit, Plotly & Lottie</h6>", unsafe_allow_html=True)
