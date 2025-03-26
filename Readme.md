# Advanced Emotion & Depression Analysis

An advanced application integrating deep learning (DL), machine learning (ML), and natural language processing (NLP) techniques to analyze emotions and assess depression risk. Built using Streamlit for an interactive dashboard, Plotly for dynamic visualizations, and TensorFlow/Keras for the DL model.



## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture & Design](#architecture--design)
  - [System Design Diagram (SDD)](#system-design-diagram-sdd)
- [Project Components](#project-components)
  - [Dashboard](#dashboard)
  - [Deep Learning Model Integration](#deep-learning-model-integration)
  - [Research Report & Documentation](#research-report--documentation)
- [Model Preparation](#model-preparation)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Future Work](#future-work)




## Project Overview

This project offers a holistic approach to emotion and depression analysis by combining traditional ML techniques with DL and NLP. It collects user responses via an interactive questionnaire, computes normalized emotion scores, and predicts depression risk using both a weighted scoring algorithm and a neural network. The project also provides detailed visualizations and a downloadable report, making it suitable for both personal insights and academic research.



## Key Features

- **Interactive Questionnaire:** Collects data on various emotional states via intuitive Streamlit sliders.
- **Emotion Scoring:** Normalizes responses and calculates weighted emotion scores.
- **Depression Risk Prediction:**
  - **Weighted Method:** Custom risk assessment based on key emotional indicators.
  - **Neural Network:** Utilizes a Keras model (pre-trained or built on the fly) to simulate depression risk predictions.
- **Dynamic Visualizations:** Displays results using Plotly charts (bar, pie, and line graphs).
- **NLP Integration:** Processes textual input for sentiment analysis to provide additional insights.
- **Custom UI:** Incorporates custom CSS styling and animated elements (via Lottie) for an engaging user experience.



## Architecture & Design

The project is structured into multiple functional modules that work together to provide a comprehensive analysis. The architecture leverages data collection, feature engineering, model inference, and data visualization components.

### System Design Diagram (SDD)

Below is a text-based flow diagram outlining the project’s design:

```bash
+---------------------+
| User Interaction    |
| (Streamlit UI)      |
+----------+----------+
           |
           v
+---------------------+      +--------------------------+
| Data Collection     |----->| Emotion Questionnaire    |
| & Preprocessing     |      | (Slider Inputs, Text NLP)|
+---------------------+      +--------------------------+
           |
           v
+---------------------+      +--------------------------+
| Feature Engineering |----->| Emotion Score Calculation|
| (Normalization,     |      | & Weighted Depression    |
| Weight Assignment)  |      | Risk Computation         |
+---------------------+      +--------------------------+
           |
           v
+---------------------+
| Model Integration   |
| (Deep Learning &    |
| ML Prediction)      |
+---------------------+
           |
           v
+---------------------+
| Visualization &     |
| Reporting           |
| (Plotly Charts, CSV |
| Download, Research  |
| Documentation)      |
+---------------------+

```

## Project Components

### Dashboard

- **Questionnaire Section:** 
  - Users adjust sliders to rate different emotional states (e.g., Joy, Sadness, Anxiety, etc.).
  - Responses are normalized and used to calculate emotion scores.
  
- **Visualization:**
  - **Bar Chart & Pie Chart:** Visualize emotion distribution and scores.
  - **Line Chart:** Simulates weekly depression risk trends.
  
- **Personalized Feedback:**
  - Based on calculated weighted depression risk, the dashboard offers tailored advice and warnings.

### Deep Learning Model Integration

- **DL Model Loading:**
  - Tries to load a pre-trained Keras model (`model.h5`). If unavailable, it builds a simple neural network architecture.
  
- **Sentiment Analysis:**
  - Provides an NLP interface where users can enter text describing their emotional state.
  - Simulated inference gives a depression risk percentage based on text analysis.

### Research Report & Documentation

- **Project Documentation:**
  - Explains methodology, data collection, and feature engineering.
  - Details the integration of DL/ML/NLP components for depression risk prediction.
  
- **Code Architecture Overview:**
  - Presents a clear code flow, highlighting each functional part from data collection to final reporting.
  - The project serves as a foundational framework for further academic research and real-world applications.



## Model Preparation

- **Neural Network Architecture:**
  - **Input Layer:** Accepts a feature vector of six emotion scores.
  - **Hidden Layers:** Includes Dense layers with ReLU activation and Dropout for regularization.
  - **Output Layer:** Uses a sigmoid function to predict depression risk as a probability.
  
- **Compilation:**
  - The model is compiled with the Adam optimizer and binary cross-entropy loss.
  
- **Deployment:**
  - Either a pre-trained model is loaded for inference or a fresh model is constructed if none is available.
  
- **Integration with NLP:**
  - For textual inputs, the DL model can be integrated with NLP pipelines for preprocessing text and mapping sentiment to risk metrics.



## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-depression-analysis.git
   cd emotion-depression-analysis
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Ensure that you have Python 3.7+ and required packages like `streamlit`, `plotly`, `pandas`, `numpy`, `requests`, and `tensorflow` installed*.

3. **Run the Application::**
   ```bash
   streamlit run app.py
   ```


## Usage
- **Dashboard:** Navigate through the questionnaire, adjust the sliders, and visualize results.

- **Deep Learning Model:**  Input text to simulate sentiment analysis and depression risk prediction.

- **Research Report:** Access detailed project documentation and download reports for further analysis.

## Future Work

- **Model Training:** Use clinical data to train the DL model for improved accuracy.

- **Enhanced NLP Processing:** Integrate advanced NLP techniques for better sentiment analysis.

- **User Personalization:** Incorporate longitudinal tracking and historical data analysis.

- **Real-Time Analytics:** Implement continuous monitoring with IoT integration.

---
 *Stay strong and take care!*