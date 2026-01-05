from model_server import ModelServer
from load_balancer import LoadBalancer

# create model servers
server_1 = ModelServer("model_server_1")
server_2 = ModelServer("model_server_2")
server_3 = ModelServer("model_server_3")

servers = [server_1, server_2, server_3]
lb = LoadBalancer(servers)

# simulate traffic
print("Starting distributed inference simulation...\n")

for i in range(15):
    try:
        if i == 8:
            print(" model_server_2 has crashed\n")
            server_2.is_active = False

        result = lb.route_request(input_data={"sample": i})

        print(
            f"Request {i+1} | "
            f"Server: {result['server']} | "
            f"Prediction: {result['prediction']} | "
            f"Latency: {result['latency']}s"
        )

    except Exception as e:
        print(" Request failed:", e)

print("\n Server Stats:")
for s in servers:
    print(f"{s.name} → Requests served: {s.requests_served}")
