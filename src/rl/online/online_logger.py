import csv
import json
from pathlib import Path


class OnlineLogger:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def log_step(self, data: dict) -> None:
        self.rows.append(data)

    def save(self) -> None:
        if not self.rows:
            return

        fieldnames = sorted({key for row in self.rows for key in row.keys()})

        with open(self.save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({
                    key: json.dumps(value) if isinstance(value, dict) else value
                    for key, value in row.items()
                })