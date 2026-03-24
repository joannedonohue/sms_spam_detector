# SMS Spam Classifier

A production-ready NLP pipeline that classifies SMS messages as spam or legitimate in real time — deployed via a Gradio web interface.

---

## Overview

This project builds an end-to-end SMS spam detection system using TF-IDF vectorization and a LinearSVC classifier, wrapped in a scikit-learn Pipeline for clean, deployable inference. The model is served through a Gradio interface, enabling real-time classification without requiring any ML infrastructure.

---

## Pipeline Architecture

```
Raw SMS Text
     │
TF-IDF Vectorizer  (text → numerical features)
     │
LinearSVC Classifier  (spam vs. ham)
     │
Gradio Web Interface  (real-time predictions)
```

The entire preprocessing + classification flow is encapsulated in a single scikit-learn `Pipeline` object, making the model portable and easy to deploy.

---

## Model Performance

Trained and evaluated on the UCI SMS Spam Collection (5,574 messages, 13.4% spam prevalence):

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Ham (legitimate) | 0.99 | 1.00 | 0.99 |
| Spam | 0.97 | 0.92 | 0.94 |

---

## Dataset

- **Source:** UCI SMS Spam Collection
- **Size:** ~5,574 labeled messages
- **Class distribution:** 86.6% ham / 13.4% spam

---

## Tech Stack

| Component | Tool |
|---|---|
| Text vectorization | TF-IDF (scikit-learn) |
| Classifier | LinearSVC |
| Pipeline | scikit-learn Pipeline |
| Deployment | Gradio |
| Data handling | pandas |
| Language | Python |

---

## Repository Structure

```
sms-spam-classifier/
├── sms_spam_detector.ipynb              # Model training & evaluation
└── README.md
```

---

## Outcomes

- Achieved **0.99 F1-score on ham** and **0.94 F1-score on spam** detection
- Built a clean, single-pipeline architecture that handles preprocessing and classification in one step
- Deployed via Gradio for live, browser-based inference with no backend infrastructure required
- Demonstrated practical NLP deployment patterns applicable to production messaging systems

---

## Getting Started

```bash
pip install scikit-learn gradio pandas
jupyter notebook sms_spam_detector.ipynb
```

To launch the Gradio interface:

```bash
python gradio_sms_text_classification.py
```
