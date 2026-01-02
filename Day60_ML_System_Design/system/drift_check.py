def check_drift(train_mean, live_mean, threshold=0.2):
    drift_score = abs(train_mean - live_mean) / train_mean
    print(f" Drift score: {round(drift_score, 2)}")

    return drift_score > threshold
