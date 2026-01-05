class SLAMonitor:
    def __init__(self, latency_threshold=1.0):
        self.latency_threshold = latency_threshold
        self.violations = 0

    def check_latency(self, latency):
        if latency > self.latency_threshold:
            self.violations += 1
            return False
        return True
