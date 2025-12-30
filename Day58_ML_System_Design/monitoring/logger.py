from datetime import datetime

def log_request(input_data, output):
    with open("monitoring/requests.log", "a") as f:
        f.write(f"{datetime.now()} | INPUT={input_data} | OUTPUT={output}\n")
