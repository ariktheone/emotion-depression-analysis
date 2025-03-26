import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Emotion & Depression Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
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

# Title
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>🧠 Depression & Behavioral Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Analyzing Mental Health Through Emotions & Behavioral Patterns</h4>", unsafe_allow_html=True)
st.markdown("---")

# **Emotion & Behavior Analysis Questions**
questions = {
    "Happiness": "How often do you feel happy?",
    "Calmness": "Do you feel emotionally stable?",
    "Confidence": "Do you believe in yourself?",
    "Sadness": "Do you experience prolonged sadness?",
    "Anxiety": "Do you feel nervous frequently?",
    "Frustration": "How often do you feel irritated?",
    "Loneliness": "Do you feel disconnected from others?",
    "Guilt": "Do you blame yourself excessively?",
    "Fatigue": "Do you feel mentally & physically exhausted?",
    "Work-Life Balance": "Can you manage work & personal life?",
    "Socialization": "How often do you engage socially?",
    "Cultural Influence": "Does culture shape your emotions?",
}

responses = {}
cols = st.columns(3)

# **User Inputs**
for idx, (emotion, question) in enumerate(questions.items()):
    with cols[idx % 3]:  
        responses[emotion] = st.slider(question, min_value=0, max_value=10, value=5, format="%d")

# **Depression Risk Calculation**  
if st.button("Analyze Results"):
    st.markdown("---")
    
    emotion_scores = {key: val / 10 for key, val in responses.items()}
    df = pd.DataFrame(emotion_scores.items(), columns=['Emotion', 'Score']).sort_values(by="Score", ascending=False)

    # **Depression Risk Score Calculation (Weighted Factors)**
    depression_score = (
        (10 - responses["Happiness"]) * 0.3 + 
        responses["Sadness"] * 0.25 + 
        responses["Anxiety"] * 0.2 + 
        responses["Loneliness"] * 0.15 + 
        responses["Fatigue"] * 0.1
    ) * 10  # Convert to percentage

    # **Depression Gauge Chart**
    fig_depression = go.Figure(go.Indicator(
        mode="gauge+number",
        value=depression_score,
        title={'text': "Depression Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))

    # **Depression Risk Alerts**
    if depression_score >= 75:
        st.markdown("<div class='alert-box'>🚨 High Depression Risk Detected! Seek Support Immediately.</div>", unsafe_allow_html=True)
    elif 50 <= depression_score < 75:
        st.warning("⚠️ Moderate Depression Risk. Consider consulting a mental health professional.")
    else:
        st.success("✅ Low Depression Risk. Maintain a healthy lifestyle & emotional well-being.")

    # **Cultural Influence Score & Gauge Chart**
    cultural_impact = responses["Cultural Influence"] * 10
    fig_culture = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cultural_impact,
        title={'text': "Cultural Influence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FFD700"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))

    # **Emotion Correlation Heatmap**
    emotion_matrix = pd.DataFrame(np.random.rand(len(questions), len(questions)), 
                                  columns=list(questions.keys()), 
                                  index=list(questions.keys()))
    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(emotion_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)

    # **Radar Chart**
    categories = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values.append(values[0])  
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Emotions'))
    fig_radar.update_layout(title="Emotional Pattern Radar Chart")

    # **Violin Plot - Emotion Distribution**
    fig_violin, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=[list(emotion_scores.values())], ax=ax)
    ax.set_xticklabels(["Emotion Distribution"])

    # **Display Visuals**
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📊 Depression Risk Analysis")
        st.plotly_chart(fig_depression, use_container_width=True)
    with col2:
        st.subheader("🌍 Cultural Impact Analysis")
        st.plotly_chart(fig_culture, use_container_width=True)

    st.subheader("📉 Advanced Behavioral Analysis")
    col3, col4 = st.columns([1, 1])
    with col3:
        st.subheader("📊 Emotion Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)
    with col4:
        st.subheader("🔥 Emotion Heatmap")
        st.pyplot(fig_heatmap)

    # **Data Table**
    st.subheader("📜 Detailed Emotion Report")
    st.dataframe(df)

    # **Behavioral Insights**
    st.subheader("🧠 Behavioral Patterns & Mental Health Risks")
    if responses["Loneliness"] > 7 and responses["Socialization"] < 3:
        st.warning("🚨 High Loneliness Detected! Consider engaging in social activities.")
    if responses["Work-Life Balance"] < 4:
        st.warning("⚠️ Poor Work-Life Balance! Take time for yourself.")
    if depression_score > 70:
        st.error("🚨 Severe Depression Risk! Immediate action recommended.")

    st.markdown("<h3 style='text-align: center; color: #FFD700;'>Take care of your mental health! 💙</h3>", unsafe_allow_html=True)
    st.markdown("---")
    