from model import DummyModel
from cache import PredictionCache
from monitor import SLAMonitor

model = DummyModel()
cache = PredictionCache()
monitor = SLAMonitor()

def fallback_prediction(x):
    return 0  # Safe default prediction

def run_inference(input_data):
    key = tuple(input_data)

    cached = cache.get(key)
    if cached:
        return cached, "cache"

    try:
        prediction, latency = model.predict(input_data)

        if not monitor.check_latency(latency):
            return fallback_prediction(input_data), "fallback_slow"

        cache.set(key, prediction)
        return prediction, "model"

    except Exception:
        return fallback_prediction(input_data), "fallback_error"
