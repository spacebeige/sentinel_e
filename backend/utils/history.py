import json
import os
from datetime import datetime

class HistoryLogger:
    def __init__(self, history_file="data/history/user_history.json"):
        self.history_file = history_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.history_file):
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump([], f)

    def log_interaction(self, input_type, input_path_or_text, result):
        """
        Log the user interaction (Input -> System -> Result).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "input_source": input_path_or_text if len(str(input_path_or_text)) < 200 else "Text Input (...)",
            "system_result": result
        }
        
        try:
            with open(self.history_file, 'r+') as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to log history: {e}")

    def get_history(self):
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except:
            return []
