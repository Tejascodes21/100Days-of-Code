import random

def route_request(input_data, old_model, new_model, canary_ratio=0.2):
    if random.random() < canary_ratio:
        return "canary", new_model.predict(input_data)[0]
    return "stable", old_model.predict(input_data)[0]
