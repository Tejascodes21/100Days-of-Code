import random
import time

class ModelServer:
    def __init__(self, name, is_active=True):
        self.name = name
        self.is_active = is_active
        self.requests_served = 0

    def predict(self, input_data):
        if not self.is_active:
            raise Exception(f"{self.name} is DOWN")

        # simulate latency
        latency = random.uniform(0.05, 0.3)
        time.sleep(latency)

        self.requests_served += 1

        # dummy prediction
        prediction = random.choice([0, 1])

        return {
            "server": self.name,
            "prediction": prediction,
            "latency": round(latency, 3)
        }
