import csv
import pathlib
import re

class Logger:
    """Handles logging of session data."""

    def __init__(self, log_path: pathlib.Path):
        self.log_path = log_path

    def append_log_entry_csv(self, entry: dict):
        """Append an entry to a CSV file, create header if new file."""
        file_exists = self.log_path.exists()
        try:
            with self.log_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)
        except Exception as e:
            print(f"[LOG][ERROR] Could not write to {self.log_path}: {e}")

    @staticmethod
    def safe_slug(text: str, max_len: int = 60) -> str:
        """Make a filesystem-safe short slug from text (for filenames)."""
        text = re.sub(r"\s+", "_", text.strip())
        text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
        return text[:max_len] if text else "query"