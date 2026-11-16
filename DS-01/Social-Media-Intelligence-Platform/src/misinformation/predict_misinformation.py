
import os
import sys
import joblib

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "misinformation_classifier.joblib"
)

model = joblib.load(MODEL_PATH)

def predict_misinformation(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a valid statement.")

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities)

    result = (
        "Potentially misleading"
        if prediction == 0
        else "Likely reliable"
    )

    return {
        "statement": text,
        "prediction": result,
        "confidence": round(float(confidence), 4),
        "warning": (
            "This is a model prediction, not a verified fact-check."
        )
    }


if __name__ == "__main__":

    print("MISINFORMATION DETECTION SYSTEM")
    print("=" * 45)

    if len(sys.argv) > 1:
        statement = " ".join(sys.argv[1:])
    else:
        statement = input("\nEnter a statement: ")

    try:
        result = predict_misinformation(statement)

        print("\nPrediction:", result["prediction"])
        print("Model confidence:", result["confidence"])
        print("Warning:", result["warning"])

    except ValueError as error:
        print("Error:", error)
