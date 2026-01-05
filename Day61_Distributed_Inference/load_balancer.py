import random

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers

    def route_request(self, input_data):
        active_servers = [s for s in self.servers if s.is_active]

        if not active_servers:
            raise Exception("No active model servers available")

        server = random.choice(active_servers)
        return server.predict(input_data)

                