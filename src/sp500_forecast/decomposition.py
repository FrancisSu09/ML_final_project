from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

from .config import DecompositionConfig


EPS = 1.0e-12


@dataclass(slots=True)
class Component:
    name: str
    values: np.ndarray


@dataclass(slots=True)
class DecompositionResult:
    imfs: list[np.ndarray]
    residual: np.ndarray
    vmd_modes: list[np.ndarray]
    vmd_k: int
    vmd_alpha: float
    pso_fitness: float
    components: list[Component]


def decompose_signal(
    signal: np.ndarray,
    config: DecompositionConfig,
    *,
    target: str | None = None,
) -> DecompositionResult:
    """ICEEMDAN initial decomposition, PSO-VMD re-decomposition of IMF1."""
    values = np.asarray(signal, dtype=float)
    if config.mode == "none":
        return DecompositionResult(
            imfs=[],
            residual=np.zeros_like(values),
            vmd_modes=[],
            vmd_k=0,
            vmd_alpha=0.0,
            pso_fitness=0.0,
            components=[Component("Original", values)],
        )

    imfs_arr, residual = iceemdan(
        values,
        max_imfs=config.max_imfs,
        ensembles=config.ensembles,
        noise_strength=config.noise_strength,
        sift_max_iter=config.sift_max_iter,
        sift_tolerance=config.sift_tolerance,
        random_seed=config.random_seed,
    )
    imfs = [imfs_arr[i].copy() for i in range(imfs_arr.shape[0])]
    if not imfs:
        imfs = [values - np.mean(values)]
        residual = np.full_like(values, np.mean(values), dtype=float)

    if config.mode == "iceemdan":
        components = [Component(f"IMF{idx}", imf) for idx, imf in enumerate(imfs, start=1)]
        components.append(Component("Res", residual))
        return DecompositionResult(
            imfs=imfs,
            residual=residual,
            vmd_modes=[],
            vmd_k=0,
            vmd_alpha=0.0,
            pso_fitness=0.0,
            components=components,
        )
    if config.mode != "iceemdan_pso_vmd":
        raise ValueError(
            "Unsupported decomposition.mode. Use 'iceemdan_pso_vmd', 'iceemdan', or 'none'."
        )

    best_k, best_alpha, best_score = choose_vmd_params(imfs[0], config, target=target)
    vmd_modes_arr = vmd(
        imfs[0],
        k=best_k,
        alpha=best_alpha,
        tau=config.vmd_tau,
        tolerance=config.vmd_tolerance,
        max_iter=config.vmd_max_iter,
    )
    vmd_modes = [vmd_modes_arr[i].copy() for i in range(vmd_modes_arr.shape[0])]

    components: list[Component] = []
    for idx, mode in enumerate(vmd_modes, start=1):
        components.append(Component(f"VIMF{idx}", mode))
    for idx, imf in enumerate(imfs[1:], start=2):
        components.append(Component(f"IMF{idx}", imf))
    components.append(Component("Res", residual))

    return DecompositionResult(
        imfs=imfs,
        residual=residual,
        vmd_modes=vmd_modes,
        vmd_k=best_k,
        vmd_alpha=best_alpha,
        pso_fitness=best_score,
        components=components,
    )


def iceemdan(
    signal: np.ndarray,
    *,
    max_imfs: int,
    ensembles: int,
    noise_strength: float,
    sift_max_iter: int,
    sift_tolerance: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-contained ICEEMDAN-style decomposition following the paper equations."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("ICEEMDAN expects a one-dimensional signal")
    if ensembles < 1:
        raise ValueError("ensembles must be >= 1")
    if max_imfs < 1:
        raise ValueError("max_imfs must be >= 1")

    rng = np.random.default_rng(random_seed)
    n = len(x)
    noise_modes = np.zeros((ensembles, max_imfs, n), dtype=float)
    for i in range(ensembles):
        noise = rng.standard_normal(n)
        modes, _ = emd(
            noise,
            max_imfs=max_imfs,
            sift_max_iter=sift_max_iter,
            sift_tolerance=sift_tolerance,
        )
        count = min(max_imfs, len(modes))
        if count:
            for mode_idx in range(count):
                mode = modes[mode_idx]
                noise_modes[i, mode_idx] = mode / max(float(np.std(mode)), EPS)

    beta0 = noise_strength * max(float(np.std(x)), EPS)
    first_means = np.vstack(
        [
            emd_local_mean(
                x + beta0 * noise_modes[i, 0],
                sift_max_iter=sift_max_iter,
                sift_tolerance=sift_tolerance,
            )
            for i in range(ensembles)
        ]
    )
    residue = np.mean(first_means, axis=0)
    imfs: list[np.ndarray] = [x - residue]

    for mode_index in range(1, max_imfs):
        if count_extrema(residue) < 2:
            break
        beta = noise_strength * max(float(np.std(residue)), EPS)
        means = np.vstack(
            [
                emd_local_mean(
                    residue + beta * noise_modes[i, mode_index],
                    sift_max_iter=sift_max_iter,
                    sift_tolerance=sift_tolerance,
                )
                for i in range(ensembles)
            ]
        )
        next_residue = np.mean(means, axis=0)
        imf = residue - next_residue
        if float(np.std(imf)) <= EPS:
            break
        imfs.append(imf)
        residue = next_residue

    return np.asarray(imfs, dtype=float), residue


def emd(
    signal: np.ndarray,
    *,
    max_imfs: int,
    sift_max_iter: int,
    sift_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mode decomposition used by the ICEEMDAN implementation."""
    residue = np.asarray(signal, dtype=float).copy()
    imfs: list[np.ndarray] = []

    for _ in range(max_imfs):
        if count_extrema(residue) < 2:
            break
        imf = sift_imf(
            residue,
            max_iter=sift_max_iter,
            tolerance=sift_tolerance,
        )
        if float(np.std(imf)) <= EPS:
            break
        imfs.append(imf)
        residue = residue - imf

    if not imfs:
        return np.empty((0, len(residue)), dtype=float), residue
    return np.asarray(imfs, dtype=float), residue


def emd_local_mean(
    signal: np.ndarray,
    *,
    sift_max_iter: int,
    sift_tolerance: float,
) -> np.ndarray:
    imf = sift_imf(signal, max_iter=sift_max_iter, tolerance=sift_tolerance)
    return np.asarray(signal, dtype=float) - imf


def sift_imf(signal: np.ndarray, *, max_iter: int, tolerance: float) -> np.ndarray:
    h = np.asarray(signal, dtype=float).copy()
    for _ in range(max_iter):
        maxima, minima = extrema_indices(h)
        if len(maxima) + len(minima) < 2:
            break
        upper = interpolate_envelope(h, maxima)
        lower = interpolate_envelope(h, minima)
        mean_envelope = (upper + lower) / 2.0
        previous = h.copy()
        h = h - mean_envelope
        change = float(np.sum((previous - h) ** 2) / (np.sum(previous**2) + EPS))
        if is_imf(h, mean_envelope, tolerance=tolerance) or change < tolerance:
            break
    return h


def is_imf(values: np.ndarray, mean_envelope: np.ndarray, *, tolerance: float) -> bool:
    maxima, minima = extrema_indices(values)
    extrema_count = len(maxima) + len(minima)
    zero_crossings = int(np.sum(values[:-1] * values[1:] < 0))
    extrema_condition = abs(extrema_count - zero_crossings) <= 1
    mean_condition = float(np.mean(np.abs(mean_envelope))) <= tolerance * (
        float(np.mean(np.abs(values))) + EPS
    )
    return extrema_condition and mean_condition


def extrema_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    diff = np.diff(values)
    signs = np.sign(diff)
    signs = _fill_zero_signs(signs)
    maxima = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1
    minima = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0] + 1
    return maxima.astype(int), minima.astype(int)


def count_extrema(values: np.ndarray) -> int:
    maxima, minima = extrema_indices(values)
    return len(maxima) + len(minima)


def interpolate_envelope(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if len(points) == 0:
        return np.full(n, float(np.mean(values)), dtype=float)

    points = np.unique(np.concatenate(([0], points, [n - 1]))).astype(int)
    y = values[points]
    grid = np.arange(n)
    if len(points) >= 4:
        return CubicSpline(points, y, bc_type="natural")(grid)
    return np.interp(grid, points, y)


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    signs = signs.copy()
    if len(signs) == 0:
        return signs
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(len(signs) - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]
    signs[signs == 0] = 1
    return signs


def vmd(
    signal: np.ndarray,
    *,
    k: int,
    alpha: float,
    tau: float,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    """Variational mode decomposition following the standard VMD ADMM routine.

    The implementation mirrors the public reference algorithm used by vmdpy:
    mirror extension, analytic positive-frequency spectrum, iterative ADMM
    updates, then removal of the mirrored boundaries. This is substantially
    closer to Dragomiretskiy and Zosso's VMD than a direct full-spectrum FFT
    shortcut and is important for reproducible re-decomposition of IMF1.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(x) < 4:
        return np.tile(x, (k, 1)) / k

    original_length = len(x)
    if original_length % 2:
        x = x[:-1]
    half = len(x) // 2
    mirrored = np.concatenate([np.flip(x[:half]), x, np.flip(x[-half:])])
    total_length = len(mirrored)
    freqs = np.arange(total_length, dtype=float) / total_length - 0.5 - (1.0 / total_length)

    alpha_arr = float(alpha) * np.ones(k, dtype=float)
    f_hat = np.fft.fftshift(np.fft.fft(mirrored))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: total_length // 2] = 0

    u_hat_plus = np.zeros((max_iter, total_length, k), dtype=np.complex128)
    omega_plus = np.zeros((max_iter, k), dtype=float)
    if k > 1:
        omega_plus[0] = 0.5 / k * np.arange(k)
    lambda_hat = np.zeros((max_iter, total_length), dtype=np.complex128)

    u_diff = tolerance + EPS
    n = 0
    sum_uk = np.zeros(total_length, dtype=np.complex128)
    while u_diff > tolerance and n < max_iter - 1:
        sum_uk = u_hat_plus[n, :, k - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, 0] = (
            f_hat_plus - sum_uk - lambda_hat[n] / 2.0
        ) / (1.0 + alpha_arr[0] * (freqs - omega_plus[n, 0]) ** 2)
        if k > 1:
            _update_omega(omega_plus, u_hat_plus, freqs, n, 0, total_length)

        for mode_idx in range(1, k):
            sum_uk = (
                u_hat_plus[n + 1, :, mode_idx - 1]
                + sum_uk
                - u_hat_plus[n, :, mode_idx]
            )
            u_hat_plus[n + 1, :, mode_idx] = (
                f_hat_plus - sum_uk - lambda_hat[n] / 2.0
            ) / (1.0 + alpha_arr[mode_idx] * (freqs - omega_plus[n, mode_idx]) ** 2)
            _update_omega(omega_plus, u_hat_plus, freqs, n, mode_idx, total_length)

        lambda_hat[n + 1] = lambda_hat[n] + tau * (
            np.sum(u_hat_plus[n + 1], axis=1) - f_hat_plus
        )
        n += 1

        u_diff = EPS
        for mode_idx in range(k):
            delta = u_hat_plus[n, :, mode_idx] - u_hat_plus[n - 1, :, mode_idx]
            u_diff += (1.0 / total_length) * np.vdot(delta, delta)
        u_diff = float(np.abs(u_diff))

    final_iter = max(n, 1)
    u_hat = np.zeros((total_length, k), dtype=np.complex128)
    u_hat[total_length // 2 :, :] = u_hat_plus[final_iter, total_length // 2 :, :]
    u_hat[total_length // 2 : 0 : -1, :] = np.conj(
        u_hat_plus[final_iter, total_length // 2 :, :]
    )
    u_hat[0, :] = np.conj(u_hat[-1, :])

    modes = np.zeros((k, total_length), dtype=float)
    for mode_idx in range(k):
        modes[mode_idx] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, mode_idx])))
    modes = modes[:, total_length // 4 : 3 * total_length // 4]
    if original_length % 2:
        modes = np.pad(modes, ((0, 0), (0, 1)), mode="edge")
    modes = modes[:, :original_length]
    modes[-1] += np.asarray(signal, dtype=float).reshape(-1)[:original_length] - np.sum(modes, axis=0)
    return modes


def _update_omega(
    omega_plus: np.ndarray,
    u_hat_plus: np.ndarray,
    freqs: np.ndarray,
    n: int,
    mode_idx: int,
    total_length: int,
) -> None:
    spectrum = np.abs(u_hat_plus[n + 1, total_length // 2 :, mode_idx]) ** 2
    denom = float(np.sum(spectrum))
    if denom > EPS:
        omega_plus[n + 1, mode_idx] = float(
            np.dot(freqs[total_length // 2 :], spectrum) / denom
        )
    else:
        omega_plus[n + 1, mode_idx] = omega_plus[n, mode_idx]


def choose_vmd_params(
    signal: np.ndarray,
    config: DecompositionConfig,
    *,
    target: str | None = None,
) -> tuple[int, float, float]:
    if config.use_paper_vmd_params and target in config.paper_vmd_params:
        params = config.paper_vmd_params[target]
        k = int(params["k"])
        alpha = float(params["alpha"])
        return k, alpha, vmd_fitness(signal, k=k, alpha=alpha, config=config)
    return pso_optimize_vmd(signal, config)


def pso_optimize_vmd(signal: np.ndarray, config: DecompositionConfig) -> tuple[int, float, float]:
    rng = np.random.default_rng(config.random_seed)
    particles = max(1, config.pso_particles)
    positions = np.column_stack(
        [
            rng.uniform(config.vmd_k_min, config.vmd_k_max, size=particles),
            rng.uniform(config.vmd_alpha_min, config.vmd_alpha_max, size=particles),
        ]
    )
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_scores = np.array([_vmd_fitness_for_position(pos, signal, config) for pos in positions])
    best_idx = int(np.argmin(personal_scores))
    global_best = personal_best[best_idx].copy()
    global_score = float(personal_scores[best_idx])

    lower = np.array([config.vmd_k_min, config.vmd_alpha_min], dtype=float)
    upper = np.array([config.vmd_k_max, config.vmd_alpha_max], dtype=float)

    for _ in range(config.pso_iterations):
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            config.pso_inertia * velocities
            + config.pso_cognitive * r1 * (personal_best - positions)
            + config.pso_social * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        for idx, pos in enumerate(positions):
            score = _vmd_fitness_for_position(pos, signal, config)
            if score < personal_scores[idx]:
                personal_scores[idx] = score
                personal_best[idx] = pos
                if score < global_score:
                    global_score = float(score)
                    global_best = pos.copy()

    best_k = int(round(float(global_best[0])))
    best_k = int(np.clip(best_k, config.vmd_k_min, config.vmd_k_max))
    best_alpha = float(np.clip(global_best[1], config.vmd_alpha_min, config.vmd_alpha_max))
    return best_k, best_alpha, global_score


def vmd_fitness(signal: np.ndarray, *, k: int, alpha: float, config: DecompositionConfig) -> float:
    modes = vmd(
        signal,
        k=k,
        alpha=alpha,
        tau=config.vmd_tau,
        tolerance=config.vmd_tolerance,
        max_iter=config.vmd_max_iter,
    )
    entropy = float(np.mean([envelope_entropy(mode) for mode in modes]))
    reconstruction_error = float(
        np.linalg.norm(np.asarray(signal, dtype=float) - np.sum(modes, axis=0))
        / (np.linalg.norm(signal) + EPS)
    )
    if config.pso_fitness == "envelope_entropy":
        return entropy
    if config.pso_fitness == "reconstruction_entropy":
        return entropy + 10.0 * reconstruction_error
    if config.pso_fitness == "reconstruction_error":
        return reconstruction_error
    raise ValueError(f"Unsupported pso_fitness: {config.pso_fitness}")


def envelope_entropy(mode: np.ndarray) -> float:
    envelope = np.abs(hilbert(np.asarray(mode, dtype=float)))
    total = float(np.sum(envelope))
    if total <= EPS:
        return 0.0
    probability = envelope / total
    probability = probability[probability > EPS]
    return float(-np.sum(probability * np.log(probability)) / np.log(len(envelope)))


def _vmd_fitness_for_position(
    position: np.ndarray,
    signal: np.ndarray,
    config: DecompositionConfig,
) -> float:
    k = int(round(float(position[0])))
    k = int(np.clip(k, config.vmd_k_min, config.vmd_k_max))
    alpha = float(np.clip(position[1], config.vmd_alpha_min, config.vmd_alpha_max))
    return vmd_fitness(signal, k=k, alpha=alpha, config=config)
