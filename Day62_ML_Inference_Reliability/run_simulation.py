from inference import run_inference
import random

def generate_request():
    return [random.randint(0, 10) for _ in range(5)]

if __name__ == "__main__":
    for i in range(20):
        request = generate_request()
        prediction, source = run_inference(request)
        print(f"Request {i+1} | Prediction: {prediction} | Served by: {source}")
