import os
from datetime import datetime


class Logger:
    def __init__(self, log_file="train.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"=== Log started at {datetime.now()} ===\n")

    def log(self, msg):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{timestamp} {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
