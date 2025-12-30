from pipelines.data_ingestion import ingest_data
from pipelines.feature_pipeline import build_features
from pipelines.training_pipeline import train_model
from serving.inference_service import predict
from monitoring.logger import log_request

# Offline pipeline
raw_df = ingest_data()
feature_df = build_features(raw_df)
train_model(feature_df)

# Online inference simulation
sample_request = {
    "age": 30,
    "income": 60000,
    "credit_score": 710,
    "income_per_age": 2000
}

result = predict(sample_request)
log_request(sample_request, result)

print(" Prediction Result:", result)
