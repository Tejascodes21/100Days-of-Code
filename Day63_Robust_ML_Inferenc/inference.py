from model import DummyClassifier
from validator import validate_input
from confidence import is_confident

model = DummyClassifier()

def fallback_decision():
    return {
        "prediction": "REVIEW_REQUIRED",
        "reason": "Low confidence or invalid input"
    }

def predict(features):
    valid, message = validate_input(features)
    if not valid:
        return {
            "prediction": "REJECTED",
            "reason": message
        }

    prediction, confidence = model.predict_proba(features)

    if not is_confident(confidence):
        return fallback_decision()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3)
    }
