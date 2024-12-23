from flask import Flask, request, jsonify, render_template
import re
import joblib
import unicodedata
import indicnlp.tokenize.indic_tokenize as indic_tokenize
import langid
import numpy as np

app = Flask(__name__)

# Load models and vectorizers for all languages
MODELS = {
    "english": {
        "model": joblib.load("sentiment_analysis_model_eng.pkl"),
        "vectorizer": joblib.load("tfidf_vectorizer_eng.pkl"),
    },
    "hindi": {
        "model": joblib.load("xgb_model_hin.pkl"),
        "vectorizer": joblib.load("tfidf_vectorizer_hin.pkl"),
    },
}

# Sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Stopwords for Hindi
manual_hindi_stopwords = {
    "के", "का", "कि", "की", "है", "को", "पर", "यह", "से", "में", "और",
    "एक", "था", "जो", "तक", "ने", "हो", "हैं", "लिए", "कर", "दिया", "इस", "भी",
    "तो", "ही", "नहीं", "आप", "हम", "उन", "अगर", "या", "जब", "तक", "मुझे", "हमारे"
}

# Preprocessing functions
def clean_text(text, language="english"):
    """Basic text cleaning and stopword removal."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()

    if language == "hindi":
        text = " ".join([word for word in text.split() if word not in manual_hindi_stopwords])

    return text

def preprocess_text(text, language):
    """Advanced preprocessing based on language."""
    if language == "hindi":
        text = re.sub(r"[^\u0900-\u097F\s]", "", text)  # Keep only Hindi characters
        text = clean_text(text, language)
        text = " ".join(indic_tokenize.trivial_tokenize(text))  # Tokenize using Indic NLP
    else:
        text = clean_text(text, language)  # Default preprocessing for English

    return text

def predict_sentiment(text, language):
    """Predict sentiment for the given text and language."""
    try:
        # Get the model and vectorizer for the selected language
        model = MODELS[language]["model"]
        vectorizer = MODELS[language]["vectorizer"]

        # Preprocess the text
        cleaned_text = preprocess_text(text, language)

        # Vectorize the cleaned text
        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict sentiment and probabilities
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        result = sentiment_mapping.get(prediction, "Unknown")
        probability_dict = {
            "Negative": round(float(probabilities[0]) * 100, 2),
            "Positive": round(float(probabilities[1]) * 100, 2) if len(probabilities) > 2 else None,
            "Neutral": round(float(probabilities[-1]) * 100, 2),
        }

        return result, probability_dict
    except Exception as e:
        return f"Error: {str(e)}", None

def is_language_valid_alt(text, language):
    """
    Alternative implementation using langid library.
    """
    if not text.strip():
        print("Error: Text is empty or whitespace.")
        return False

    detected_language, confidence = langid.classify(text)
    print(f"Detected language: {detected_language}, Confidence: {confidence}")

    if language.lower() == "hindi" and detected_language != "hi":
        return False
    elif language.lower() == "english" and detected_language != "en":
        return False
    return True

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        text = data.get("text", "")
        language = data.get("language", "english")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if language not in MODELS:
            return jsonify({"error": "Unsupported language"}), 400

        # Validate text language
        if not is_language_valid_alt(text, language):
            return jsonify({"error": f"Input text does not match the selected language ({language})."}), 400

        # Predict sentiment for the review text
        result, probabilities = predict_sentiment(text, language)

        if probabilities is None:
            return jsonify({"error": "Error in prediction"}), 500

        return jsonify({
            "sentiment": result,
            "probabilities": probabilities,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
