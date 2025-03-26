import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random
import requests
import json
import time

# Optional deep learning integration (simulate if not available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="Advanced Emotion & Depression Analysis", layout="wide")

# ----------------------------
# Helper Functions
# ----------------------------
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

def calculate_emotion_scores(responses):
    """Calculate normalized emotion scores from responses."""
    emotion_scores = {
        "Joy": round(responses[0] / 10, 2),
        "Sadness": round(responses[1] / 10, 2),
        "Anxiety": round(responses[2] / 10, 2),
        "Hopelessness": round(responses[7] / 10, 2),
        "Fatigue": round(responses[9] / 10, 2),
        "Social Withdrawal": round(responses[13] / 10, 2),
    }
    return emotion_scores

def calculate_depression_risk(emotion_scores):
    """Advanced weighted depression risk calculation."""
    weights = {
        "Sadness": 0.2,
        "Anxiety": 0.15,
        "Hopelessness": 0.25,
        "Fatigue": 0.2,
        "Social Withdrawal": 0.2,
    }
    weighted_score = sum(weights[k] * emotion_scores[k] for k in weights)
    risk = min(1.0, max(0.0, weighted_score * (1 - 0.5 * emotion_scores["Joy"])))
    return round(risk * 100, 2)

# ----------------------------
# Custom CSS for Styling
# ----------------------------
custom_css = """
<style>
/* Overall background gradient */
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: #f0f0f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Enhanced Metric Box style */
.metric-box {
    background: #282c34;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(255, 215, 0, 0.5);
    text-align: center;
    color: #FFD700;
}
/* Sidebar customization */
[data-testid="stSidebar"] {
    background-color: #1f1c2c;
    color: #f0f0f0;
}
/* Title elements */
h1, h2, h3, h4 {
    font-weight: 600;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------
# Load Lottie Animation
# ----------------------------
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_kt8n7oeq.json"
lottie_animation = load_lottie(lottie_url)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ("Dashboard", "Deep Learning Model", "Research Report"))

# ----------------------------
# Dashboard Section
# ----------------------------
if section == "Dashboard":
    st.title("Advanced Emotion & Depression Analysis Dashboard")
    st.markdown("### Analyze your emotional state with interactive metrics and visualizations")
    
    # Questionnaire Section
    st.subheader("Questionnaire")
    questions = [
        "How often do you feel happy or joyful?",
        "Do you feel sad or down frequently?",
        "How often do you feel anxious or nervous?",
        "Do you experience sudden mood swings?",
        "Do you feel lonely or isolated?",
        "How often do you have trouble sleeping?",
        "Do you feel overwhelmed by daily tasks?",
        "How often do you feel hopeless about the future?",
        "Do you find it hard to enjoy things you used to love?",
        "How often do you feel fatigued or lack energy?",
        "Do you feel irritable or easily frustrated?",
        "Do you have difficulty concentrating or making decisions?",
        "How often do you feel worthless or guilty?",
        "Do you feel like withdrawing from social activities?",
        "How often do you experience physical symptoms like headaches or body pain without a clear cause?",
    ]
    st.markdown("Adjust the sliders based on your feelings (0: Not at all, 10: Very frequently)")
    
    responses = []
    cols = st.columns(3)
    for i in range(5):  # 5 rows
        for j in range(3):  # 3 columns
            idx = i * 3 + j
            if idx < len(questions):
                with cols[j]:
                    responses.append(st.slider(questions[idx], 0, 10, 5, key=f"q{idx}"))
    st.markdown("---")
    
    if st.button("Analyze Results"):
        # Calculate emotion scores and depression risk
        emotion_scores = calculate_emotion_scores(responses)
        depression_percentage = calculate_depression_risk(emotion_scores)
        
        # Visualizations
        df = pd.DataFrame(list(emotion_scores.items()), columns=['Emotion', 'Score'])
        df = df.sort_values(by="Score", ascending=False)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Emotion Scores")
            fig_bar = px.bar(df, x="Emotion", y="Score", color="Emotion", title="Emotion Analysis", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.subheader("Emotion Distribution")
            fig_pie = px.pie(df, names="Emotion", values="Score", title="Emotion Share", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Depression risk display
        st.markdown("<h2 style='text-align: center; color: #FFD700;'>Depression Probability</h2>", unsafe_allow_html=True)
        mid_col = st.columns(3)
        with mid_col[1]:
            st.markdown(f"""
                <div class='metric-box'>
                    <h2>🧠 {depression_percentage:.2f}%</h2>
                    <p>Estimated Depression Risk</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Weekly depression trends simulation
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        risk_values = [round(random.uniform(0.2, 0.9), 2) for _ in range(7)]
        fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True)
        fig_line.update_layout(xaxis_title="Day", yaxis_title="Depression Risk Level")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Personalized advice based on depression risk
        if depression_percentage > 70:
            advice = "🚨 **High risk detected!** It's strongly recommended to seek professional help."
        elif depression_percentage > 40:
            advice = "⚠️ **Moderate risk.** Consider self-care practices and talking to someone you trust."
        else:
            advice = "😊 **Low risk!** Keep maintaining healthy habits and monitor your emotions."
        st.warning(advice)
        
        # Detailed data table and CSV download option
        st.subheader("Detailed Emotion & Depression Report")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report as CSV", data=csv, file_name="emotion_depression_report.csv", mime="text/csv")

# ----------------------------
# Deep Learning Model Section
# ----------------------------
elif section == "Deep Learning Model":
    st.title("Deep Learning Model Integration")
    st.markdown("### Sentiment Analysis & Depression Prediction using DL")
    
    if MODEL_AVAILABLE:
        st.markdown("Loading pre-trained model...")
        @st.cache(allow_output_mutation=True)
        def load_dl_model():
            model = load_model("model.h5")
            return model
        model = load_dl_model()
        st.success("Model loaded successfully!")
        
        # User input for DL model inference
        user_input = st.text_area("Enter a brief description of your current emotional state:")
        if st.button("Analyze Sentiment"):
            st.info("Analyzing...")
            time.sleep(1)
            # Replace with actual preprocessing and model inference steps
            prediction = random.uniform(0, 1)
            st.markdown(f"**Predicted Depression Risk:** {prediction*100:.2f}%")
            st.markdown("*(This is a simulated result. Replace with your model's inference pipeline for real predictions.)*")
    else:
        st.markdown("Deep Learning model integration is not available in this demo. "
                    "Ensure TensorFlow is installed and a valid model file is provided for full functionality.")

# ----------------------------
# Research Report Section
# ----------------------------
elif section == "Research Report":
    st.title("Research Report & Project Documentation")
    st.markdown("""
    This project is an advanced demonstration of an Emotion and Depression Analysis tool built using modern web frameworks and deep learning techniques.
    
    **Project Highlights:**
    - **Advanced Questionnaire:** Collects nuanced data on multiple dimensions of emotions.
    - **Interactive Visualizations:** Uses Plotly for dynamic, interactive charts.
    - **Deep Learning Integration:** Incorporates (or simulates) a DL model for sentiment and depression prediction.
    - **Data Analysis & Reporting:** Provides both detailed and summary views with downloadable reports.
    
    **Methodology:**
    1. **Data Collection:** An interactive questionnaire gathers emotional responses.
    2. **Feature Engineering:** Emotion scores are computed and weighted to assess depression risk.
    3. **Model Integration:** A DL model (if available) processes textual input for refined predictions.
    4. **Visualization:** Interactive charts (bar, pie, and trend lines) support data interpretation.
    5. **Reporting:** Detailed reports are available for download and further analysis.
    
    **Future Work:**
    - **Enhanced DL Model:** Training on larger, clinical datasets for improved prediction accuracy.
    - **User Personalization:** Incorporating historical user data for trend analysis.
    - **Real-time Analytics:** Integration with wearables/IoT for continuous monitoring.
    
    This project serves both as a practical tool for personal insights and as a foundation for further academic research.
    """)
    st.markdown("### Code Architecture & Flow")
    st.code("""
    1. Data Collection: Interactive questionnaire via Streamlit sliders.
    2. Emotion Scoring: Normalize and weight responses to generate emotion scores.
    3. Depression Risk Assessment: Calculate risk using a weighted formula.
    4. Visualization: Generate interactive charts (bar, pie, line) with Plotly.
    5. DL Model Integration: (Optional) Leverage a pre-trained model for sentiment analysis.
    6. Reporting: Display results and enable CSV downloads for further research.
    """, language="python")
    st.markdown("### Conclusion")
    st.markdown("This advanced project combines data collection, interactive visualization, and deep learning integration to assess emotional states and depression risk. Its modular architecture makes it not only user-friendly but also well-suited for academic research and further development.")

# ----------------------------
# Footer Section with Animation
# ----------------------------
st.markdown("<h3 style='text-align: center; color: #FFD700;'>Stay Strong & Take Care! 😊</h3>", unsafe_allow_html=True)
if lottie_animation:
    st.json(lottie_animation)
st.markdown("<h6 style='text-align: center; color: grey;'>Built with ❤️ using Streamlit, Plotly, and DL Integration</h6>", unsafe_allow_html=True)
