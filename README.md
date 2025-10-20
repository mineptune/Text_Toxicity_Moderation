Text Toxicity Moderation Model with Gradio Interface
Overview

This project demonstrates how to detect toxic comments (e.g., hate speech, insults, profanity, harassment) using Natural Language Processing (NLP) techniques in a Jupyter Notebook environment.
It combines traditional machine learning (TF-IDF + Logistic Regression) with modern transformer models (BERT) and provides an interactive Gradio interface for real-time text or voice-based moderation.

🧭 Project Objectives

Develop and evaluate a Text Toxicity Classification Model.

Compare baseline and transformer-based approaches.

Visualize dataset distributions and model performance.

Deploy an interactive demo using Gradio with both text and speech-to-text inputs.

⚙️ Features

✅ Two modeling paths:

Baseline: TF-IDF Vectorizer + Logistic Regression

Advanced: Pre-trained unitary/toxic-bert transformer

✅ Interactive Interface (Gradio):

Type or speak a sentence.

Real-time classification (Toxic / Non-Toxic).

Displays confidence score and transcript if spoken input is used.

✅ Explainable Notebook Workflow:

Clean, modular sections with Markdown explanations.

Visual EDA plots and detailed evaluation metrics.

Ready for fine-tuning or deployment expansion.

🧩 Project Structure
Text_Toxicity_Moderation_Model_With_Gradio.ipynb
├── 1. Introduction
├── 2. Setup and Imports
├── 3. Load Dataset
├── 4. Exploratory Data Analysis (EDA)
├── 5. Data Preprocessing
├── 6. Baseline Model (TF-IDF + Logistic Regression)
├── 7. Advanced Model (Pre-trained BERT)
├── 8. Model Evaluation & Sample Predictions
├── 9. Conclusion & Future Work
└── 10. Interactive Gradio Interface (Text + Voice)

🧰 Requirements

Install all dependencies before running the notebook:

pip install pandas numpy scikit-learn matplotlib seaborn transformers torch gradio SpeechRecognition


For voice input to work, you’ll need a microphone and SpeechRecognition installed with a compatible backend (Google Speech API is used by default).

▶️ How to Run

Open the Notebook

Launch Jupyter Notebook or Google Colab.

Upload the file Text_Toxicity_Moderation_Model_With_Gradio.ipynb.

Run All Cells

Execute each cell sequentially.

The Gradio interface will appear at the bottom of the notebook.

Use the Interface

Type or speak any sentence.

View real-time predictions and confidence levels.

🧠 Model Details

Baseline Model:

TF-IDF features (max 5,000 terms)

Logistic Regression classifier

Simple yet strong performance on small datasets.

Transformer Model:

unitary/toxic-bert pre-trained on toxic comment data.

Context-aware and multilingual to some extent.

Supports direct inference without fine-tuning.

📊 Evaluation Metrics

Accuracy

Precision / Recall / F1-Score

Confusion Matrix Visualization

The notebook provides performance summaries for both traditional and transformer-based models.

🎙️ Gradio Interface

The final cell launches a browser-based interface that enables:

Real-time toxicity detection.

Speech-to-text transcription (using SpeechRecognition).

Clean UX for experimentation and demos.

🔐 Ethical & Responsible AI Notes

Toxicity detection models are prone to:

Biases (language, dialect, cultural nuance).

False positives/negatives in sarcasm or informal text.

Ensure ongoing retraining, fairness audits, and dataset diversity when deploying in production.

🔮 Future Enhancements

Fine-tune BERT on a domain-specific dataset.

Add multi-label classification (toxic, severe_toxic, obscene, etc.).

Integrate explainability (e.g., SHAP or LIME).

Wrap as an API for use in moderation pipelines.

👨‍💻 Author
Neptune Michel
AI Solution Architect
LinkedIn: https://www.linkedin.com/in/neptunemichel
