import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

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

def detect_emotions(text):
    """Detect multiple emotions from text with proper probability distribution."""
    
    # Tokenization with truncation & padding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Inference (disable gradient calculation for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probability distribution
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Get top 3 emotions (more realistic for complex emotions)
    top_indices = np.argsort(scores)[-3:][::-1]  # Sort and get top 3
    top_emotions = [(EMOTION_LABELS[i], scores[i]) for i in top_indices if scores[i] > 0.1]  # Filter low confidence

    # Print all emotion scores for debugging
    print("\n🔹 Emotion Confidence Scores:")
    for emotion, score in zip(EMOTION_LABELS, scores):
        print(f"{emotion}: {score:.4f}")

    return top_emotions

# Example Usage
if __name__ == "__main__":
    text = input("Enter a sentence: ")
    detected_emotions = detect_emotions(text)
    
    print("\n🎭 Detected Emotions:")
    for emotion, confidence in detected_emotions:
        print(f" - {emotion} (Confidence: {confidence:.2f})")
