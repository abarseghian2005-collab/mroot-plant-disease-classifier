"""Print saved evaluation results without loading images or models."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = json.loads((root / "reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json").read_text())
print(json.dumps(result, indent=2))
