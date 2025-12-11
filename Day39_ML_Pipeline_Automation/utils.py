# utils.py
import os
import joblib
import pandas as pd
import logging
from datetime import datetime

def ensure_dirs(cfg):
    for d in [cfg["models_dir"], cfg["artifacts_dir"], cfg["logs_dir"]]:
        os.makedirs(d, exist_ok=True)

def read_config(path="config.yaml"):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(logfile):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger()

def save_model(model, scaler, cfg, score_summary):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{cfg['model_prefix']}_{ts}.pkl"
    model_path = os.path.join(cfg["models_dir"], model_name)
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    return model_name, model_path

def log_metrics(cfg, model_name, metrics: dict):
    file = os.path.join(cfg["artifacts_dir"], "metrics.csv")
    row = {"timestamp": pd.Timestamp.now(), "model_name": model_name}
    row.update(metrics)
    df_new = pd.DataFrame([row])
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(file, index=False)

def read_data(path):
    return pd.read_csv(path)
