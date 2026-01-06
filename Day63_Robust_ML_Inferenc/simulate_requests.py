from inference import predict

test_inputs = [
    [10, 20, 30, 40, 50],       # valid
    [5, 200, 10, 5, 2],         # out-of-range
    ["a", 2, 3, 4, 5],          # malicious input
    [1, 2, 3],                  # invalid length
    [15, 25, 35, 45, 55],       # valid
]

print(" Starting robust inference simulation...\n")

for i, data in enumerate(test_inputs, start=1):
    result = predict(data)
    print(f"Request {i} | Input: {data} | Output: {result}")
