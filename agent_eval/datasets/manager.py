"""Dataset management — CRUD, versioning, and persistence for eval datasets."""

import json
import os
from pathlib import Path
from agent_eval.core.types import Trace, Label, Scenario


DEFAULT_DATASET_DIR = Path(os.environ.get(
    "AGENT_EVAL_DATASET_DIR",
    str(Path(__file__).parent / "scenarios"),
))


class DatasetManager:
    """Manage evaluation datasets: scenarios, traces, and labels."""

    def __init__(self, dataset_dir: Path | None = None):
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def save_scenarios(self, scenarios: list[Scenario], version: str = "v1"):
        """Save scenarios to a versioned JSON file."""
        version_dir = self.dataset_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        filepath = version_dir / "scenarios.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in scenarios], f, indent=2)

    def load_scenarios(self, version: str = "v1") -> list[Scenario]:
        """Load scenarios from a versioned JSON file."""
        filepath = self.dataset_dir / version / "scenarios.json"
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Scenario(
                scenario_id=s["scenario_id"],
                query=s["query"],
                description=s.get("description", ""),
            )
            for s in data
        ]

    def save_traces(self, traces: list[Trace], version: str = "v1"):
        """Save traces to a versioned JSON file."""
        version_dir = self.dataset_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        filepath = version_dir / "traces.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in traces], f, indent=2)

    def load_traces(self, version: str = "v1") -> list[Trace]:
        """Load traces from a versioned JSON file."""
        filepath = self.dataset_dir / version / "traces.json"
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Trace.from_dict(t) for t in data]

    def save_labels(self, labels: dict[str, Label], version: str = "v1"):
        """Save labels to a versioned JSON file."""
        version_dir = self.dataset_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        filepath = version_dir / "labels.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in labels.items()},
                f,
                indent=2,
            )

    def load_labels(self, version: str = "v1") -> dict[str, Label]:
        """Load labels from a versioned JSON file."""
        filepath = self.dataset_dir / version / "labels.json"
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: Label.from_dict(v) for k, v in data.items()}

    def version_exists(self, version: str) -> bool:
        """Check if a dataset version exists."""
        return (self.dataset_dir / version).exists()
