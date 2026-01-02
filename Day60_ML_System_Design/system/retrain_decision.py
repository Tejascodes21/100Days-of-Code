def should_retrain(drift_detected, accuracy):
    if drift_detected:
        print(" Drift detected — retraining required")
        return True

    if accuracy < 0.7:
        print(" Accuracy below threshold — retraining required")
        return True

    print(" Model performing within limits")
    return False
