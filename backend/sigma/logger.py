"""
Sigma Logger: Forensic logging for experimental analysis.
Captures hypotheses, boundaries, safety assessments, metrics.
"""

import json
import os
from datetime import datetime
from common.utils import generate_id, get_timestamp

class SigmaLogger:
    def __init__(self, output_dir="sentinel_history"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    async def log_run(self, run_data: dict):
        """
        Persists forensic log to disk (and later DB).
        Includes boundary analysis, hypotheses, safety reports.
        """
        run_id = generate_id()
        timestamp = get_timestamp()
        filename = f"sentinel_sigma_run_{timestamp}_{run_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        final_record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "data": run_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(final_record, f, indent=2)
            
        # If boundary violations exist, log severity metrics
        if "boundary_analysis" in run_data:
            boundary_data = run_data["boundary_analysis"]
            import logging
            logger = logging.getLogger("Sigma-Logger")
            logger.info(f"Run {run_id}: Boundary Severity={boundary_data.get('cumulative_severity', 0)} "
                       f"Violations={boundary_data.get('violation_count', 0)}")
            
        return run_id
