import time
import random

class DummyModel:
    def predict(self, x):
        # Simulate variable latency
        latency = random.uniform(0.1, 1.5)
        time.sleep(latency)

        # Simulate random failure
        if random.random() < 0.1:
            raise RuntimeError("Model crashed")

        return int(sum(x) % 2), latency
