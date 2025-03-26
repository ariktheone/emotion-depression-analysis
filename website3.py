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
st.markdown("""
    <h1 style='text-align: center; color: #00D4FF;'>Emotion & Depression Analysis Dashboard</h1>
    <h4 style='text-align: center; color: #FFD700;'>Real-time emotion detection with deep insights</h4>
    """, unsafe_allow_html=True)

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
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Emotion Confidence Levels")
        fig_bar = px.bar(df, x="Confidence", y="Emotion", color="Emotion", orientation='h',
                         title="Emotion Confidence Scores", text_auto=True, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🎭 Emotion Distribution")
        fig_pie = px.pie(df, names="Emotion", values="Confidence", title="Emotion Share", hole=0.4, height=400,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Depression Risk Display
    st.subheader("📈 Depression Probability Analysis")
    st.markdown(f"🧠 **Estimated Depression Probability:** `{depression_risk * 100:.2f}%`")
    
    # Weekly Trends (Smoothed Data)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    risk_values = np.clip(np.cumsum(np.random.normal(0, 0.05, 7)) + depression_risk, 0, 1)
    
    fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True,
                        labels={"x": "Day", "y": "Depression Risk Level"},
                        color_discrete_sequence=['red'])
    fig_line.update_traces(mode='lines+markers', line=dict(width=3))
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Histogram of Confidence Levels
    st.subheader("📊 Confidence Level Distribution")
    fig_hist = px.histogram(df, x="Confidence", nbins=10, title="Emotion Confidence Distribution",
                            color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Box Plot for Emotion Confidence Spread
    st.subheader("📦 Emotion Confidence Spread")
    fig_box = px.box(df, y="Confidence", title="Emotion Confidence Variance",
                     color_discrete_sequence=['#AB63FA'])
    st.plotly_chart(fig_box, use_container_width=True)

    # Detailed Data Table
    st.subheader("📜 Detailed Emotion & Depression Report")
    st.dataframe(df)
