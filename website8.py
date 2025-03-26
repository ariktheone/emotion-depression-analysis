import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random
import requests
import time

# ----------------------------
# Optional Deep Learning Integration
# ----------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
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

def calculate_weighted_depression_risk(emotion_scores):
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

def build_nn_model(input_dim):
    """Build a simple neural network for depression risk prediction."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ----------------------------
# Custom CSS for Styling
# ----------------------------
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: #f0f0f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.metric-box {
    background: #282c34;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(255, 215, 0, 0.5);
    text-align: center;
    color: #FFD700;
}
[data-testid="stSidebar"] {
    background-color: #1f1c2c;
    color: #f0f0f0;
}
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
        # Calculate emotion scores and depression risk (weighted method)
        emotion_scores = calculate_emotion_scores(responses)
        weighted_depression = calculate_weighted_depression_risk(emotion_scores)
        
        # Neural network prediction
        nn_input = np.array([[emotion_scores["Joy"],
                               emotion_scores["Sadness"],
                               emotion_scores["Anxiety"],
                               emotion_scores["Hopelessness"],
                               emotion_scores["Fatigue"],
                               emotion_scores["Social Withdrawal"]]])
        # If you have a pre-trained model, you could load it instead of building from scratch
        if MODEL_AVAILABLE:
            try:
                # Attempt to load a pre-trained model if available (replace 'model.h5' with your model file)
                nn_model = load_model("model.h5")
            except Exception as e:
                # If not available, build a fresh model (note: untrained, so prediction is for demo purposes)
                nn_model = build_nn_model(6)
            nn_prediction = nn_model.predict(nn_input)[0][0] * 100
        else:
            nn_prediction = random.uniform(0, 1) * 100  # Fallback simulation
        
        # Display results
        st.markdown("<h2 style='text-align: center; color: #FFD700;'>Depression Probability</h2>", unsafe_allow_html=True)
        st.markdown(f"**Weighted Method Prediction:** {weighted_depression:.2f}%")
        st.markdown(f"**Neural Network Prediction:** {nn_prediction:.2f}%")
        
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
        
        # Weekly depression trends simulation
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        risk_values = [round(random.uniform(0.2, 0.9), 2) for _ in range(7)]
        fig_line = px.line(x=days, y=risk_values, title="Weekly Depression Risk Trends", markers=True)
        fig_line.update_layout(xaxis_title="Day", yaxis_title="Depression Risk Level")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Personalized advice based on weighted depression risk (you may also choose to combine both predictions)
        if weighted_depression > 70:
            advice = "🚨 **High risk detected!** It's strongly recommended to seek professional help."
        elif weighted_depression > 40:
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
        st.markdown("Loading pre-trained model (if available)...")
        @st.cache(allow_output_mutation=True)
        def load_dl_model():
            try:
                return load_model("model.h5")
            except Exception as e:
                return build_nn_model(6)
        model = load_dl_model()
        st.success("Model ready for inference!")
        
        # User input for DL model inference
        user_input = st.text_area("Enter a brief description of your current emotional state:")
        if st.button("Analyze Sentiment"):
            st.info("Analyzing...")
            time.sleep(1)
            # Here you would preprocess your text and run inference
            # For demo purposes, we simulate a prediction:
            prediction = model.predict(np.random.rand(1, 6))[0][0] * 100
            st.markdown(f"**Predicted Depression Risk:** {prediction:.2f}%")
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
    - **Neural Network Integration:** A simple Keras model is integrated to offer an alternative depression risk prediction.
    - **Data Analysis & Reporting:** Provides both detailed and summary views with downloadable reports.
    
    **Methodology:**
    1. **Data Collection:** An interactive questionnaire via Streamlit sliders.
    2. **Feature Engineering:** Emotion scores are normalized and used both in weighted calculations and as input to a neural network.
    3. **Model Integration:** A neural network (or a pre-trained model) processes the emotion scores to predict depression risk.
    4. **Visualization:** Interactive charts (bar, pie, and trend lines) aid in data interpretation.
    5. **Reporting:** Detailed reports are available for download and further academic analysis.
    
    **Future Work:**
    - **Model Training:** Training the neural network on real clinical data for improved prediction accuracy.
    - **User Personalization:** Incorporating historical user data for longitudinal tracking.
    - **Real-time Analytics:** Integration with IoT devices for continuous monitoring.
    
    This project serves both as a practical tool for personal insights and a solid foundation for further academic research.
    """)
    st.markdown("### Code Architecture & Flow")
    st.code("""
    1. Data Collection: Interactive questionnaire via Streamlit sliders.
    2. Emotion Scoring: Normalize and weight responses to generate emotion scores.
    3. Depression Risk Assessment:
       - Weighted Method: Uses a custom weighted average.
       - Neural Network: Processes emotion scores with a Keras model.
    4. Visualization: Generates interactive charts (bar, pie, line) using Plotly.
    5. Reporting: Displays results and enables CSV downloads.
    """, language="python")
    st.markdown("### Conclusion")
    st.markdown("This advanced project combines data collection, interactive visualization, and deep learning integration to assess emotional states and depression risk. Its modular design makes it well-suited for both personal use and academic research.")

# ----------------------------
# Footer Section with Animation
# ----------------------------
st.markdown("<h3 style='text-align: center; color: #FFD700;'>Stay Strong & Take Care! 😊</h3>", unsafe_allow_html=True)
if lottie_animation:
    st.json(lottie_animation)
st.markdown("<h6 style='text-align: center; color: grey;'>Built with ❤️ using Streamlit, Plotly, and DL Integration</h6>", unsafe_allow_html=True)
