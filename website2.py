import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random
from transformers import pipeline

# Load Emotion Detection Model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Dashboard Styling
st.set_page_config(page_title="Emotion & Depression Analysis", layout="wide")

# UI Layout
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>Emotion & Depression Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Real-time emotion detection with depression probability</h4>", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.header("🔍 Enter Your Text")
user_input = st.sidebar.text_area("Type a sentence to analyze emotions", "")

if user_input:
    # Emotion Detection
    result = emotion_classifier(user_input)
    emotions = {entry['label']: entry['score'] for entry in result[0]}
    
    # Depression Risk Calculation (Randomized for demo, replace with ML model if needed)
    depression_risk = round(random.uniform(0.2, 0.9) * (1 - emotions.get('joy', 0)), 2)
    
    # Convert to DataFrame
    df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
    df = df.sort_values(by="Confidence", ascending=False)

    # Main Dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Emotion Confidence Levels")
        fig_bar = px.bar(df, x="Emotion", y="Confidence", color="Emotion", title="Emotion Confidence Scores", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🎭 Emotion Distribution")
        fig_pie = px.pie(df, names="Emotion", values="Confidence", title="Emotion Share", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Depression Risk Display
    st.subheader("📈 Depression Probability Analysis")
    st.markdown(f"🧠 **Estimated Depression Probability:** `{depression_risk * 100:.2f}%`")

    # Depression Trends (Dummy Data)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    risk_values = [random.uniform(0.2, 0.9) for _ in range(7)]
    
    fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True)
    fig_line.update_layout(xaxis_title="Day", yaxis_title="Depression Risk Level")
    st.plotly_chart(fig_line, use_container_width=True)

    # Detailed Data Table
    st.subheader("📜 Detailed Emotion & Depression Report")
    st.dataframe(df)

# Footer
st.markdown("<h6 style='text-align: center; color: grey;'>Built with ❤️ using Streamlit & Plotly</h6>", unsafe_allow_html=True)
