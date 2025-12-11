# pipeline.py
import os
from utils import read_config, ensure_dirs, setup_logger, read_data, save_model, log_metrics
from train import train_model
from evaluate import evaluate_model

def run_pipeline(config_path="config.yaml"):
    cfg = read_config(config_path)
    ensure_dirs(cfg)
    log_file = os.path.join(cfg["logs_dir"], "pipeline.log")
    logger = setup_logger(log_file)
    logger.info("Starting pipeline run")

    
    df = read_data(cfg["data_path"])
    logger.info(f"Loaded data: {cfg['data_path']} rows={len(df)}")

    # Train
    model, scaler, X_test, y_test = train_model(df, cfg["features"], cfg["target_col"], cfg)
    logger.info("Training completed")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Evaluation metrics: {metrics}")

    
    model_name, model_path = save_model(model, scaler, cfg, metrics)
    logger.info(f"Saved model: {model_name} -> {model_path}")

    # Log metrics to artifacts CSV
    log_metrics(cfg, model_name, metrics)
    logger.info("Metrics logged to artifacts")

    logger.info("Pipeline run completed")

if __name__ == "__main__":
    run_pipeline()
