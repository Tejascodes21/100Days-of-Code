import pandas as pd
from serving.blue_green import load_models
from serving.canary_router import route_request

columns = ["age", "cholesterol", "blood_pressure", "bmi", "smoker", "physical_activity"]

sample_input = pd.DataFrame(
    [[45, 220, 140, 28.4, 1, 0]],
    columns=columns
)

blue_model, green_model = load_models()

route, prediction = route_request(
    sample_input,
    old_model=blue_model,
    new_model=green_model,
    canary_ratio=0.3
)

print(f"Request served by: {route} model")
print(f"Prediction: {prediction}")
