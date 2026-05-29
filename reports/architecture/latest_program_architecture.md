# Latest Program Architecture

This diagram reflects the current S&P 500 High/Low pipeline after the delta-target revision.

```mermaid
flowchart TD
    A[Investing.com S&P 500 OHLC<br/>2011-05-02 to 2023-03-31<br/>3000 rows] --> B[Data validation and time scaling<br/>T = 1 daily by default]
    B --> C[Target loop<br/>High and Low trained independently]
    B --> X[Feature builder<br/>19 causal relative/technical features]

    C --> D[Target transform: delta<br/>Delta_t = Price_t - Price_t-1]
    D --> E[Temporal split<br/>fit: 0..2160<br/>validation: 2161..2399<br/>test: 2400..2999]
    E --> F[Fit-only component layout<br/>ICEEMDAN + PSO-VMD on fit delta series]

    F --> G[ICEEMDAN]
    G --> H[IMF1]
    H --> I[PSO-VMD re-decomposition<br/>VIMF1, VIMF2, VIMF3]
    G --> J[IMF2..IMF9]
    G --> K[Residual: Res]

    I --> L[Non-Res component models]
    J --> L
    X --> Y[Feature windows<br/>rows t-3 to t-1 only]
    Y --> L
    K --> M[Residual component model<br/>component-only]

    L --> N[BiLSTM -> SAM attention -> TCN -> Dense head]
    M --> O[BiLSTM -> SAM attention -> TCN -> Dense head]
    N --> P[Predicted component deltas]
    O --> P
    P --> Q[Sum component deltas<br/>PredictedDelta_t]

    B --> R[NaivePreviousValue_t<br/>actual price at t-1]
    Q --> S[Price reconstruction<br/>PredictedPrice_t = NaivePreviousValue_t + PredictedDelta_t]
    R --> S

    S --> T[Point prediction metrics<br/>MAPE / MAE / RMSE<br/>MDM vs naive baseline]
    S --> U[One-sided signed ACI<br/>score = Actual - Predicted over rolling volatility]
    U --> V[High: upper-tail q_high<br/>HighBound = PredictedHigh + q_high * volatility]
    U --> W[Low: lower-tail q_low<br/>LowBound = PredictedLow + q_low * volatility]

    N --> Z[Model checkpoints<br/>outputs/sp500/model_weights_delta/high/*.pt<br/>outputs/sp500/model_weights_delta/low/*.pt]
    O --> Z
```

## Component Inputs

| Component type | Components | Input per timestep | Window size | Model |
|---|---|---:|---:|---|
| VMD / IMF components | `VIMF1..VIMF3`, `IMF2..IMF9` | 1 component value + 19 features = 20 | 3 | BiLSTM -> SAM -> TCN -> Dense |
| Residual component | `Res` | 1 residual value | 3 | BiLSTM -> SAM -> TCN -> Dense |

Each target currently uses 12 component models:

```text
VIMF1, VIMF2, VIMF3, IMF2, IMF3, IMF4, IMF5, IMF6, IMF7, IMF8, IMF9, Res
```

## No-Lookahead Policy

The current main config uses `decomposition.scope: walk_forward`.

For each validation/test day `t`:

```text
1. Re-decompose only historical delta values through t-1.
2. Build component windows from t-3, t-2, t-1.
3. Build feature windows from t-3, t-2, t-1.
4. Predict Delta_t.
5. Reconstruct price with the already-known Price_t-1.
6. ACI uses only past residual scores to build the bound for day t.
```

This avoids the full-sample decomposition leakage problem that appears when the complete 2011-2023 series is decomposed before the train/test split.

