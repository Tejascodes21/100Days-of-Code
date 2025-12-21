import pandas as pd

TRAIN_PATH = "data/train_reference.csv"
INFER_PATH = "data/inference_logs.csv"

THRESHOLD = 0.15  # 15% mean shift

train = pd.read_csv(TRAIN_PATH)
infer = pd.read_csv(INFER_PATH)

print("\n Drift Report")

for col in train.columns:
    train_mean = train[col].mean()
    infer_mean = infer[col].mean()

    drift_score = abs(infer_mean - train_mean) / train_mean

    status = "DRIFT" if drift_score > THRESHOLD else "OK"

    print(
        f"{col:16} | train_mean={train_mean:.2f} "
        f"| infer_mean={infer_mean:.2f} "
        f"| drift={drift_score:.2f} → {status}"
    )
