from dataclasses import dataclass

from ..configs.base_config import PrintableConfig


@dataclass
class StabilityConfig(PrintableConfig):
    """Basic stability config"""

    stability_protection: bool = True
    loss_median_window: int = 10
    anomaly_times: int = 5
    skip_steps: int = 1
    consecutive_anomalies_steps: int = 100
