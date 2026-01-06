import random

class DummyClassifier:
    def predict_proba(self, features):
        # simulate unstable model confidence
        confidence = random.uniform(0.4, 0.99)
        prediction = 1 if confidence > 0.6 else 0
        return prediction, confidence
