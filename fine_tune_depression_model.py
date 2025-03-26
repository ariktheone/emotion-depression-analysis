# fine_tune_depression_model.py
"""
Fine-Tuning a Depression Detection NLP Model
=============================================

This script downloads a depression tweets dataset from GitHub, fine-tunes a pre-trained
DistilBERT model for binary classification (depressed vs. non-depressed), and saves the
fine-tuned model and tokenizer in the directory "fine_tuned_depression_model".

The dataset is assumed to have at least two columns:
    - tweet: The text input (e.g., social media post)
    - label: Binary label (0 for non-depressed, 1 for depressed)

Run this script to create a model that you can later load in your Streamlit application.
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# ----------------------------
# 1. Download and Prepare the Dataset
# ----------------------------
# URL for a sample depression tweets dataset hosted on GitHub
dataset_url = "https://raw.githubusercontent.com/dhruvildave/Depression-Detection-using-Twitter/master/dataset/depression_tweets.csv"

# Read CSV data directly from the URL
data = pd.read_csv(dataset_url)

# Check if the dataset has a "tweet" column and rename it to "text"
if "tweet" in data.columns:
    data = data.rename(columns={"tweet": "text"})
    
# Optionally, inspect the first few rows
print("Dataset preview:")
print(data.head())

# Split data into training and testing sets (80% train, 20% test)
train_df = data.sample(frac=0.8, random_state=42)
test_df = data.drop(train_df.index)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# ----------------------------
# 2. Tokenization
# ----------------------------
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ----------------------------
# 3. Load Pre-trained Model and Fine-Tune for Depression Detection
# ----------------------------
# For binary classification, we set num_labels=2.
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# ----------------------------
# 4. Define Evaluation Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ----------------------------
# 5. Set Up Training Arguments and Trainer
# ----------------------------
training_args = TrainingArguments(
    output_dir="fine_tuned_depression_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# ----------------------------
# 6. Fine-Tune the Model
# ----------------------------
print("Starting fine-tuning...")
trainer.train()

print("Evaluating the model...")
eval_results = trainer.evaluate()
print(eval_results)

# ----------------------------
# 7. Save the Fine-Tuned Model and Tokenizer
# ----------------------------
save_directory = "fine_tuned_depression_model"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to '{save_directory}'")
