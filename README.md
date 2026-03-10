# SMS Spam Detector

Interactive SMS spam classification app built with a linear Support Vector Classification (SVC) model and deployed via a Gradio web interface. Users can type any text message and receive an instant ham/spam prediction.

---

## Demo

The app uses a trained pipeline (TF-IDF vectorizer + LinearSVC) to classify messages in real time through a Gradio UI — no setup required.

---

## Key Results

- Trained on the **UCI SMS Spam Collection** dataset (~5,500 labeled messages)
- Pipeline: **TF-IDF vectorization** → **LinearSVC classifier**
- Clean separation between ham and spam across a range of message styles and lengths
- Deployed as an interactive **Gradio app** for live inference

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

- **Model:** LinearSVC with TF-IDF feature extraction
- **Interface:** Gradio for interactive web-based inference
- **Data:** UCI SMS Spam Collection

---

## Project Structure

```
├── gradio_sms_text_classification.ipynb    # Model training + Gradio app
└── README.md
```

---

## How It Works

1. Raw SMS text is transformed using **TF-IDF** (term frequency–inverse document frequency)
2. A **LinearSVC** model trained on labeled ham/spam messages classifies the input
3. The **Gradio interface** wraps the pipeline, allowing real-time predictions via a text input box
4. Output: "ham" (legitimate) or "spam" label with the prediction displayed in the UI
