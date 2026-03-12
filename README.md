# SMS Spam Classifier

A production-ready NLP pipeline that classifies SMS messages as spam or legitimate in real time — built with a LinearSVC model, TF-IDF vectorization, and deployed via a Gradio web interface with zero setup required.

## Overview

Trained on the UCI SMS Spam Collection dataset (~5,500 labeled messages), this classifier achieves high precision across diverse message styles and lengths. The Gradio deployment allows non-technical users to test the model interactively without writing a single line of code.

## How It Works

1. **Text preprocessing**: Raw SMS text is vectorized using TF-IDF (term frequency-inverse document frequency), capturing word importance relative to the corpus
2. **Classification**: A Linear Support Vector Classifier (LinearSVC) separates ham from spam in the transformed feature space
3. **Pipeline**: Both steps are packaged as a single sklearn Pipeline for reproducibility and clean deployment

## Model Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Ham   | 0.99      | 1.00   | 0.99 |
| Spam  | 0.97      | 0.92   | 0.94 |

## Stack

Python | scikit-learn | Gradio | TF-IDF | LinearSVC | pandas

## Usage

Run: pip install -r requirements.txt
Then: python gradio_sms_text_classification.py

Open the local Gradio URL and type any SMS message to classify it instantly.

## Dataset

UCI SMS Spam Collection — 5,574 messages, 13.4% spam prevalence
