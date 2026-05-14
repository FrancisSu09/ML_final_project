from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime by load_config
    yaml = None


@dataclass(slots=True)
class DataConfig:
    ticker: str = "^GSPC"
    start: str = "2011-05-02"
    end: str = "2023-04-01"
    csv_path: str | None = None
    date_column: str = "Date"
    sample_count: int | None = 3000
    strict_paper_sample: bool = True
    allow_truncate: bool = False
    raw_output: str = "data/raw/sp500_yahoo.csv"


@dataclass(slots=True)
class ExperimentConfig:
    targets: list[str] = field(default_factory=lambda: ["High", "Low"])
    train_ratio: float = 0.8
    train_split_date: str | None = None
    validation_ratio: float = 0.1
    window_size: int = 3
    time_scale: int = 1
    random_seed: int = 42
    device: str = "auto"


@dataclass(slots=True)
class DecompositionConfig:
    mode: str = "iceemdan_pso_vmd"
    max_imfs: int = 9
    ensembles: int = 100
    noise_strength: float = 0.2
    sift_max_iter: int = 100
    sift_tolerance: float = 0.05
    random_seed: int = 42
    pso_particles: int = 10
    pso_iterations: int = 15
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5
    vmd_k_min: int = 2
    vmd_k_max: int = 6
    vmd_alpha_min: float = 500.0
    vmd_alpha_max: float = 5000.0
    vmd_tau: float = 0.0
    vmd_tolerance: float = 1.0e-7
    vmd_max_iter: int = 500
    use_paper_vmd_params: bool = True
    pso_fitness: str = "envelope_entropy"
    paper_vmd_params: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "High": {"k": 3, "alpha": 2455.0},
            "Low": {"k": 3, "alpha": 2740.0},
        }
    )


@dataclass(slots=True)
class ModelConfig:
    variant: str = "proposed"
    hidden_size: int = 64
    hidden_grid: list[int] = field(default_factory=lambda: [32, 64, 128])
    search_hyperparameters: bool = False
    epochs: int = 100
    epoch_grid: list[int] = field(default_factory=lambda: [100, 200])
    batch_size: int = 32
    batch_grid: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    learning_rate: float = 0.001
    dropout: float = 0.2
    tcn_channels: int = 32
    tcn_kernel_size: int = 2
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    patience: int = 20


@dataclass(slots=True)
class OutputConfig:
    directory: str = "outputs/sp500"
    save_component_predictions: bool = True
    save_plots: bool = True


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def _section(cls: type, raw: dict[str, Any], name: str):
    values = raw.get(name, {}) or {}
    fields = cls.__dataclass_fields__
    filtered = {k: v for k, v in values.items() if k in fields}
    return cls(**filtered)


def load_config(path: str | Path) -> AppConfig:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config files. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig(
        data=_section(DataConfig, raw, "data"),
        experiment=_section(ExperimentConfig, raw, "experiment"),
        decomposition=_section(DecompositionConfig, raw, "decomposition"),
        model=_section(ModelConfig, raw, "model"),
        outputs=_section(OutputConfig, raw, "outputs"),
    )


def resolve_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path
