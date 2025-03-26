import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
import requests

# Streamlit Page Config
st.set_page_config(page_title="Emotion & Mental Health Analysis", layout="wide")

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
            background: linear-gradient(135deg, #0d47a1, #42a5f5);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
            text-align: center;
            color: white;
            font-size: 20px;
        }
        .alert-box {
            background: #ff4b5c;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 18px;
            animation: blink 1.5s infinite;
        }
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Title & Animation
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>Emotion & Mental Health Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Analyze emotions, mental health, and detect potential depression risk</h4>", unsafe_allow_html=True)
st.json(lottie_animation)

st.markdown("---")

# **Expanded Emotions & New Questions**
questions = {
    "Joy": "How often do you feel a genuine sense of happiness or excitement?",
    "Calmness": "Do you feel emotionally stable and peaceful most of the time?",
    "Confidence": "Do you feel self-assured and positive about your capabilities?",
    "Sadness": "Do you often experience sadness that lingers for a long time?",
    "Anxiety": "Do you feel nervous, restless, or on edge frequently?",
    "Frustration": "How often do you feel annoyed or irritated over small things?",
    "Hopelessness": "Do you often feel like there’s no way forward in life?",
    "Loneliness": "Do you frequently feel disconnected from others?",
    "Guilt": "Do you blame yourself excessively for things that happen?",
    "Overwhelm": "Do you feel like responsibilities and tasks are crushing you?",
    "Fatigue": "Do you often feel physically and mentally drained?",
    "Resentment": "Do you frequently feel bitter or hold grudges?",
    "Detachment": "Do you feel emotionally numb or disconnected from reality?",
    "Self-Doubt": "Do you second-guess yourself and your decisions often?",
    "Apathy": "Do you struggle to care about things you once enjoyed?",
}

responses = {}
cols = st.columns(3)

# **Input for Questions**
for idx, (emotion, question) in enumerate(questions.items()):
    with cols[idx % 3]:  # 3 columns
        responses[emotion] = st.slider(question, min_value=0, max_value=10, value=5, format="%d")

# Submit Button
if st.button("Analyze Results"):
    st.markdown("---")

    # **Emotion & Depression Risk Calculation**
    emotion_scores = {key: val / 10 for key, val in responses.items()}

    # **Depression Probability Calculation**
    depression_factors = {
        "Sadness": 0.2, "Anxiety": 0.15, "Hopelessness": 0.25, "Fatigue": 0.2, "Apathy": 0.2, "Loneliness": 0.2
    }
    
    weighted_depression_score = sum(depression_factors[key] * emotion_scores[key] for key in depression_factors)
    depression_risk = min(1.0, max(0.0, weighted_depression_score * (1 - 0.5 * emotion_scores.get("Joy", 0))))
    depression_risk_percentage = round(depression_risk * 100, 2)

    # **Convert Scores to Dataframe**
    df = pd.DataFrame(emotion_scores.items(), columns=['Emotion', 'Score']).sort_values(by="Score", ascending=False)

    # **Radar Chart**
    categories = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values.append(values[0])  # Closing the radar chart

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Emotions'))
    fig_radar.update_layout(title="Emotional Pattern Radar Chart")

    # **Gauge Meter for Depression Risk**
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=depression_risk_percentage,
        title={'text': "Depression Risk Level"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if depression_risk_percentage > 50 else "green"}}
    ))

    # **Weekly Depression Trends**
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    risk_values = [random.uniform(0.2, 0.9) for _ in range(7)]
    fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True)
    fig_line.update_layout(xaxis_title="Day", yaxis_title="Depression Risk Level")

    # **Display Visuals**
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📊 Emotion Scores (Radar Chart)")
        st.plotly_chart(fig_radar, use_container_width=True)
    with col2:
        st.subheader("⚠️ Depression Risk (Gauge Meter)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("📈 Weekly Trends")
    st.plotly_chart(fig_line, use_container_width=True)

    # **Depression Probability Display**
    st.markdown("<h2 style='text-align: center; color: #FFD700;'>📈 Depression Probability</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(f"""
            <div class='metric-box'>
                <h2>🧠 {depression_risk_percentage:.2f}%</h2>
                <p>Estimated Depression Risk</p>
            </div>
        """, unsafe_allow_html=True)

    # **Personalized Advice & Doctor Alert**
    if depression_risk_percentage > 80:
        st.markdown("<div class='alert-box'>🔴 High Risk! Consult a doctor immediately.</div>", unsafe_allow_html=True)
    elif depression_risk_percentage > 50:
        st.warning("⚠️ Moderate to High Risk. Consider speaking to a professional.")
    else:
        st.success("😊 You have a good emotional balance!")

    # **Detailed Data Table**
    st.subheader("📜 Detailed Emotion Report")
    st.dataframe(df)

st.markdown("<h3 style='text-align: center; color: #FFD700;'>Take care of your mental health! 💙</h3>", unsafe_allow_html=True)
