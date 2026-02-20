from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd

@dataclass
class EDAArtifacts:
    bad_indices: List[int] = field(default_factory=list)
    bad_tags: List[List[str]] = field(default_factory=list)

@dataclass
class EDAMetric:
    values: Dict[str, List[float]]  # e.g., {"CH1": [0.5, 0.6], "CH2": ...}
    unit: str = ""

@dataclass
class EDAReport:
    """
    Universal standard output for any EDA Analyzer.
    Can be consumed by a GUI for plotting, or by a headless script for saving to CSV.
    """
    summary: str
    metrics: Dict[str, EDAMetric]
    visuals: List[str] = field(default_factory=lambda: ["hist", "scatter", "mapping", "inspect"])
    artifacts: EDAArtifacts = field(default_factory=EDAArtifacts)
    inspection_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)