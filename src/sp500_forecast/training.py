from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AppConfig
from .model import BiLSTMSAMTCN, SequenceDataset
from .preprocessing import MinMaxScaler1D, make_supervised, train_val_test_masks


@dataclass(slots=True)
class ComponentPrediction:
    name: str
    predicted: np.ndarray
    actual: np.ndarray
    target_indices: np.ndarray
    best_val_loss: float
    params: dict[str, int | float]
    epochs_run: int


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_component(
    *,
    name: str,
    series: np.ndarray,
    config: AppConfig,
    device: torch.device,
    split_index: int | None = None,
) -> ComponentPrediction:
    """Train BiLSTM-SAM-TCN for one decomposed component."""
    set_random_seed(config.experiment.random_seed)
    raw = np.asarray(series, dtype=float)
    if split_index is None:
        split_index = int(len(raw) * config.experiment.train_ratio)
    split_index = int(np.clip(split_index, 1, len(raw) - 1))

    scaler = MinMaxScaler1D().fit(raw[:split_index])
    scaled = scaler.transform(raw)
    x_all, y_all, target_indices = make_supervised(
        scaled,
        window_size=config.experiment.window_size,
    )
    fit_mask, val_mask, test_mask = train_val_test_masks(
        target_indices,
        series_length=len(raw),
        train_ratio=config.experiment.train_ratio,
        validation_ratio=config.experiment.validation_ratio,
        split_index=split_index,
    )

    candidates = _candidate_params(config)
    best: ComponentPrediction | None = None
    best_score = float("inf")

    for params in candidates:
        prediction = _train_with_params(
            name=name,
            x_train=x_all[fit_mask],
            y_train=y_all[fit_mask],
            x_val=x_all[val_mask],
            y_val=y_all[val_mask],
            x_test=x_all[test_mask],
            y_test=y_all[test_mask],
            target_indices=target_indices[test_mask],
            scaler=scaler,
            params=params,
            config=config,
            device=device,
        )
        if prediction.best_val_loss < best_score:
            best_score = prediction.best_val_loss
            best = prediction

    if best is None:
        raise RuntimeError(f"Unable to train component {name}")
    return best


def _candidate_params(config: AppConfig) -> list[dict[str, int | float]]:
    model_cfg = config.model
    if not model_cfg.search_hyperparameters:
        return [
            {
                "hidden_size": model_cfg.hidden_size,
                "epochs": model_cfg.epochs,
                "batch_size": model_cfg.batch_size,
                "learning_rate": model_cfg.learning_rate,
            }
        ]

    candidates = []
    for hidden_size, epochs, batch_size in itertools.product(
        model_cfg.hidden_grid,
        model_cfg.epoch_grid,
        model_cfg.batch_grid,
    ):
        candidates.append(
            {
                "hidden_size": hidden_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": model_cfg.learning_rate,
            }
        )
    return candidates


def _train_with_params(
    *,
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_indices: np.ndarray,
    scaler: MinMaxScaler1D,
    params: dict[str, int | float],
    config: AppConfig,
    device: torch.device,
) -> ComponentPrediction:
    model = BiLSTMSAMTCN(
        input_size=1,
        hidden_size=int(params["hidden_size"]),
        tcn_channels=config.model.tcn_channels,
        tcn_kernel_size=config.model.tcn_kernel_size,
        tcn_dilations=config.model.tcn_dilations,
        dropout=config.model.dropout,
        variant=config.model.variant,
    ).to(device)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(x_val, y_val),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(x_test, y_test),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate_loss(model, val_loader, loss_fn, device)
        epochs_run = epoch
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.model.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    predicted_scaled = _predict(model, test_loader, device)
    actual_scaled = y_test.reshape(-1)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(actual_scaled)

    return ComponentPrediction(
        name=name,
        predicted=predicted,
        actual=actual,
        target_indices=target_indices,
        best_val_loss=best_val,
        params=params.copy(),
        epochs_run=epochs_run,
    )


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(model(batch_x), batch_y)
            total += float(loss.item()) * len(batch_x)
            count += len(batch_x)
    return total / max(count, 1)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            pred = model(batch_x.to(device)).detach().cpu().numpy().reshape(-1)
            values.append(pred)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)
