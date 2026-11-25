
# ============================================================
# Cyber Abuse Detection using Machine Learning for Roman Urdu
# Flask Web Application
# Supported Models:
# 1. Linear SVM
# 2. Logistic Regression
# 3. Multinomial Naive Bayes
# ============================================================

import os
import re
import joblib

from flask import Flask, render_template, request

# ------------------------------------------------------------
# 1. BASE DIRECTORY
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR
)

app.config["SECRET_KEY"] = "cyber-abuse-detection-secret-key"


# ------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------

try:
    from text_utils import clean_text

except ImportError:

    def clean_text(text):
        """
        Fallback cleaning function.
        Keep your original text_utils.py for consistency
        with the preprocessing used during model training.
        """

        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text


# ------------------------------------------------------------
# 3. MODEL FILE PATHS
# ------------------------------------------------------------

MODEL_FILES = {
    "Linear SVM": "Linear_SVM(2).pkl",

    "Logistic Regression": "Logistic_Regression(2).pkl",

    "Multinomial Naive Bayes": "Multinomial_Naive_Bayes(2).pkl"
}


# ------------------------------------------------------------
# 4. LOAD ALL THREE TRAINED MODELS
# ------------------------------------------------------------

models = {}
model_errors = {}


def load_models():

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(BASE_DIR, filename)

        try:

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {filename}"
                )

            loaded_model = joblib.load(model_path)

            # Check whether the loaded model supports prediction
            if not hasattr(loaded_model, "predict"):
                raise ValueError(
                    "Loaded file is not a valid prediction model."
                )

            models[model_name] = loaded_model

            print(f"[SUCCESS] {model_name} loaded successfully.")

        except Exception as error:

            model_errors[model_name] = str(error)

            print(f"[ERROR] Could not load {model_name}: {error}")


load_models()


# ------------------------------------------------------------
# 5. CLASSIFY ROMAN URDU COMMENTS
# ------------------------------------------------------------

def predict_abuse(comment, selected_model):

    if selected_model not in models:

        raise ValueError(
            f"{selected_model} is not available. "
            f"Please check its model file."
        )

    # Apply text cleaning
    cleaned_comment = clean_text(comment)

    if not isinstance(cleaned_comment, str) or not cleaned_comment.strip():

        raise ValueError(
            "Please enter a valid Roman Urdu comment."
        )

    # Each saved Pipeline has its own TF-IDF vectorizer.
    # Therefore, pass cleaned text directly to predict().
    trained_model = models[selected_model]

    prediction = trained_model.predict([cleaned_comment])[0]

    # Original model classes are A and NA
    predicted_label = str(prediction).strip().upper()

    if predicted_label == "A":

        result = "Abusive (A)"
        category = "abusive"

    elif predicted_label == "NA":

        result = "Non-Abusive (NA)"
        category = "non-abusive"

    else:

        result = f"Unknown Prediction: {predicted_label}"
        category = "unknown"

    return result, category


# ------------------------------------------------------------
# 6. HOME ROUTE
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    category = None
    comment = ""

    selected_model = "Linear SVM"

    error = None

    if request.method == "POST":

        comment = request.form.get("comment", "").strip()

        selected_model = request.form.get(
            "model",
            "Linear SVM"
        )

        try:

            if not comment:

                raise ValueError(
                    "Please enter a Roman Urdu comment."
                )

            result, category = predict_abuse(
                comment,
                selected_model
            )

        except Exception as exception:

            error = str(exception)

    return render_template(
        "index.html",

        result=result,

        category=category,

        error=error,

        comment=comment,

        selected_model=selected_model,

        available_models=list(MODEL_FILES.keys()),

        loaded_models=list(models.keys()),

        model_errors=model_errors
    )


# ------------------------------------------------------------
# 7. HEALTH CHECK ROUTE
# ------------------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "running",
        "application": "Cyber Abuse Detection",
        "loaded_models": list(models.keys()),
        "total_models": len(models)
    }


# ------------------------------------------------------------
# 8. MODEL STATUS ROUTE
# ------------------------------------------------------------

@app.route("/models")
def model_status():

    return {
        "available_models": list(MODEL_FILES.keys()),
        "loaded_models": list(models.keys()),
        "errors": model_errors
    }


# ------------------------------------------------------------
# 9. RUN FLASK APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\n========================================")
    print(" CYBER ABUSE DETECTION SYSTEM")
    print("========================================")

    print("\nLoaded Models:")

    for model_name in models:
        print(f"[READY] {model_name}")

    if model_errors:

        print("\nModel Loading Errors:")

        for name, error in model_errors.items():
            print(f"[ERROR] {name}: {error}")

    print("\nStarting Flask Server...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
