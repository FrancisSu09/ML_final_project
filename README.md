# Risk-Aware Stock High-Low Interval Forecasting

本專案以 S&P 500 daily High / Low 為研究標的，參考 **Gong & Xing (2024), “Predicting the highest and lowest stock price indices: A combined BiLSTM-SAM-TCN deep learning model based on re-decomposition”** 的 decomposition 與 component-model 架構，並針對全樣本分解可能造成的前視偏誤，改採 walk-forward decomposition、price-change target、causal features 與 one-sided signed Adaptive Conformal Inference (ACI)，建立具風險邊界意義的 HighBound / LowBound。

## GitHub 分支說明

**`main` 分支為本專案最終繳交版本**。其他分支保留為研究過程與對照實驗紀錄。

| Branch | 用途 | 說明 |
|---|---|---|
| `main` | 最終繳交版 | 目前主要版本。包含 walk-forward decomposition、delta / price-change target、causal technical features、one-sided signed ACI、模型權重儲存/載入、主要輸出圖表與評估比較。 |
| `no_leakage` | 嚴格無前視偏誤 baseline | 保留早期 no-leakage / walk-forward 實驗版本，用於說明直接預測 price level 時可能產生的系統性偏誤與誤差上升問題。 |
| `leakage` | 文獻式復刻 / leakage 對照 | 保留較接近原文 full-sample decomposition 的復刻流程，用於對照說明全樣本分解在真實預測情境下可能高估模型表現。 |

> Note: 若 GitHub 介面顯示其他遠端預設分支，請仍以 `main` 分支內容作為報告、程式碼與輸出結果的主要依據。

## 專案方法摘要

模型流程目前保留論文的 decomposition / component-model 骨架，但預設把模型目標改成「下一期價格變化量」，再用前一日價格還原成 High / Low：

1. 將 High / Low 價格轉成 one-step price change；若 `experiment.target_transform: level`，則回到舊版直接預測價格 level。
2. 對目標序列做 ICEEMDAN 初分解，得到 `IMF1 ... IMFm` 與 `Res`。
3. 將最高頻的 `IMF1` 用 PSO 搜尋 VMD 的最佳 `K` 與 `alpha`，再做 VMD 二次分解。
4. 對每個 `VIMF`、其餘 `IMF`、`Res` 分別訓練 `BiLSTM-SAM-TCN`。
5. 將所有 component 預測的變化量線性加總成 `PredictedDelta`。
6. 用 `PredictedPrice_t = NaivePreviousValue_t + PredictedDelta_t` 還原 High 或 Low 預測。
7. 輸出 `MAPE`、`MAE`、`RMSE`、3-day moving-average naive baseline 比較、feature ablation、MDM 檢定與 ACI coverage / bound offset 指標。

## 程式架構

整體 pipeline 由 `src/sp500_forecast/pipeline.py` 串接資料讀取、特徵工程、分解、子模型訓練、預測重建與 ACI 校準。核心神經網路定義在 `src/sp500_forecast/model.py`，其 component model 流程為：

```text
component window / component + feature window
    -> BiLSTM
    -> SAM additive attention
    -> TCN
    -> dense head
    -> next component delta prediction
```

主流程如下：

```mermaid
flowchart TD
    A["Investing.com"] --> B["data.py<br/>load and validate S&P 500 OHLC data"]
    B --> C["features.py<br/>causal OHLC / technical features"]
    B --> D["target_transform = delta<br/>High/Low price change"]
    D --> E["decomposition.py<br/>ICEEMDAN"]
    E --> F["PSO-VMD on IMF1"]
    F --> G["VIMF / IMF / Res components"]
    C --> H["preprocessing.py<br/>rolling feature windows"]
    G --> I["training.py + model.py<br/>BiLSTM -> SAM -> TCN -> dense head"]
    H --> I
    I --> J["component delta predictions"]
    J --> K["pipeline.py<br/>sum deltas and reconstruct price"]
    K --> L["metrics.py<br/>MAPE / MAE / RMSE / MDM"]
    K --> M["conformal.py<br/>one-sided signed ACI"]
    M --> N["HighBound / LowBound<br/>coverage and bound offset"]
    L --> O["outputs/sp500"]
    N --> O
```

### 主要模組責任

| Module | 主要功能 |
|---|---|
| `src/sp500_forecast/cli.py` | CLI 入口，負責 `download`、`check-data`、`run`、`smoke` 等命令。 |
| `src/sp500_forecast/config.py` | 讀取 YAML 設定並整理 data / experiment / decomposition / model / conformal / outputs 參數。 |
| `src/sp500_forecast/data.py` | 讀取 Investing 或 Yahoo Finance CSV，標準化欄位並檢查論文樣本區間。 |
| `src/sp500_forecast/features.py` | 產生 causal OHLC-derived features 與 technical indicators。 |
| `src/sp500_forecast/decomposition.py` | ICEEMDAN、PSO-VMD 與 IMF / VIMF / Res 組件分解。 |
| `src/sp500_forecast/preprocessing.py` | 建立 rolling windows、target indices 與 feature windows。 |
| `src/sp500_forecast/model.py` | 定義 BiLSTM-SAM-TCN 與 ablation variants。 |
| `src/sp500_forecast/training.py` | 訓練 component forecasters、儲存 / 載入模型權重與 scaler。 |
| `src/sp500_forecast/pipeline.py` | 串接完整實驗流程，包含 walk-forward decomposition、validation selection、test prediction 與輸出。 |
| `src/sp500_forecast/conformal.py` | one-sided signed ACI、causal rolling volatility、coverage 與 bound-band 指標。 |
| `src/sp500_forecast/metrics.py` | MAPE、MAE、RMSE、improvement rate 與 modified Diebold-Mariano test。 |

## 檔案結構圖

```text
ML_test/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── sp500.yaml                         # 主要繳交版設定
├── data/
│   └── raw/
│       └── sp500_investing.csv            # 論文區間主要資料源
├── src/
│   └── sp500_forecast/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── conformal.py
│       ├── data.py
│       ├── decomposition.py
│       ├── features.py
│       ├── metrics.py
│       ├── model.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       └── training.py
├── outputs/
│   └── sp500/                              # main 分支主要實驗輸出
│       ├── predictions_high.csv
│       ├── predictions_low.csv
│       ├── summary_high.json
│       ├── summary_low.json
│       ├── fit_high.png
│       ├── fit_low.png
│       ├── model_weights_delta/            # delta 模型權重
│       ├── evaluation_comparisons/          # 三組報告比較圖與表
│       └── feature_ablation/                # 有無 features 消融圖與表
└── tests/
    ├── test_config_data.py
    ├── test_conformal.py
    ├── test_features.py
    ├── test_model.py
    ├── test_pipeline.py
    └── test_preprocessing_features.py
```

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 資料

預設設定使用論文資料源 Investing.com 匯出的 S&P 500 historical CSV，日期對齊論文的 S&P 500 區間：

- 起日：`2011-05-02`
- 迄日：`2023-03-31`
- 目標欄位：`High`、`Low`

請先到 Investing.com 的 S&P 500 Historical Data 頁面匯出 CSV，日期範圍選 `2011-05-02` 到 `2023-03-31`，並把檔案存成：

```text
data/raw/sp500_investing.csv
```

檢查資料是否符合論文樣本：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml check-data
```

CSV 至少要有日期、最高價、最低價欄位；常見的 `Date / High / Low`、`日期 / 高 / 低`、`日期 / 最高價 / 最低價` 欄名會自動辨識。預設 `strict_paper_sample: true`，日期過濾後必須剛好是 3000 筆。

如果只是想快速測流程，可以另行把 `configs/sp500.yaml` 裡的 `data.csv_path` 清空，再使用 Yahoo Finance 便利下載：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml download
```

## 執行完整模型

同時預測 High 和 Low：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both
```

目前 `configs/sp500.yaml` 預設使用較嚴格的 `decomposition.scope: walk_forward`：

- 先切出 fit / validation / test。
- 只用 fit 區間做 ICEEMDAN / VMD 分解。
- validation 與 test 不會進入 fit decomposition。
- 預設會將 OHLC 衍生的比例特徵與技術指標作為外生特徵接到 VIMF/IMF component model；預測第 `t` 天時只使用 `t-window_size` 到 `t-1` 的 feature window，不使用第 `t` 天資料。`Res` 預設維持 component-only，避免主趨勢被外生特徵拉歪。
- Grid search 會對每組候選參數訓練所有 component model，並對 validation 每一天重新分解「截至前一天」的真實目標序列；預設目標序列是 price change，再以還原後價格的 validation RMSE 選最佳參數。
- Test evaluation 使用選出的參數，對 test 每一天重新分解「截至前一天」的真實目標序列，取最新 IMF/VIMF/Res window 預測下一天變化量，再加回前一日價格成為價格預測。
- Min-Max scaler 只用真正 fitting subset 估計 min/max，不含 validation/test target。

若要跑原本不含外生特徵的 component-only baseline：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --no-features
```

若要復刻論文 Table 6 的 full-sample decomposition 流程，可改回：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --decomposition-scope full_sample
```

若要使用較快但容易長期誤差累積的 train-only recursive baseline，可使用：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --decomposition-scope train_only_recursive
```

只預測最高價：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target high
```

只預測最低價：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target low
```

跑論文的 weekly / monthly 時間尺度變體：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --time-scale 5
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --time-scale 20
```

跑論文的 S&P 500 極端市場測試切分：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --train-split-date 2020-01-01
```

跑消融模型：

```bash
# Ablation model 1: no decomposition
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --decomposition-mode none

# Ablation model 2: ICEEMDAN only, no VMD re-decomposition
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --decomposition-mode iceemdan

# Ablation variants for the neural network block
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --model-variant no_attention
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --model-variant no_tcn
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --model-variant no_bilstm
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --model-variant lstm_sam_tcn
python -m sp500_forecast.cli --config configs/sp500.yaml run --target both --model-variant bilstm_sam_cnn
```

快速檢查環境與流程：

```bash
python -m sp500_forecast.cli smoke
```

## 輸出

預設輸出到 `outputs/sp500/`：

- `predictions_high.csv` / `predictions_low.csv`：測試集日期、實際值、預測值、誤差，以及 one-sided ACI 邊界欄位。delta 模式會額外輸出 `PredictedDelta`，其中 `Predicted = NaivePreviousValue + PredictedDelta`。校準分數保留方向：`r_t = (Actual_t - Predicted_t) / Volatility_t`；High 使用上尾 `q_high` 得到 `HighBound = Predicted + q_high * Volatility_t`，Low 使用下尾 `q_low` 得到 `LowBound = Predicted + q_low * Volatility_t`。另匯出 `BoundBandCovered`，表示 Actual 是否落在 `Predicted` 與 `HighBound` / `LowBound` 之間。
- `component_predictions_high.csv` / `component_predictions_low.csv`：每個子序列的實際與預測值。
- `summary_high.json` / `summary_low.json`：MAPE、MAE、RMSE、VMD 最佳參數、每個子模型訓練參數，以及 `conformal.bound_band_coverage`，也就是 Actual 落在 `Predicted` 與目標 bound 之間的比率。
- `best_params_high.json` / `best_params_low.json`：grid search 選出的每個 IMF/VIMF/Res 最佳模型參數；下次若 `model.search_hyperparameters: false` 且 `model.use_cached_params: true`，會自動重用。
- `decomposition_components_high.csv` / `decomposition_components_low.csv`：full-sample 模式下的 IMF/VIMF/Res 分解序列。
- `decomposition_fit_components_high.csv` / `decomposition_fit_components_low.csv`：嚴格模式下只用 fit 區間得到的 IMF/VIMF/Res 分解序列。
- `validation_predictions_high.csv` / `validation_predictions_low.csv`：嚴格模式下用於 grid search 選參的 validation 預測。
- `fit_high.png` / `fit_low.png`：實際值與預測值擬合圖。
- `decomposition_high.npz` / `decomposition_low.npz`：IMF、VIMF、Res 分解結果。
- `evaluation_comparisons/`：報告用三組比較圖與表，包含 `naive 3 天平均 vs model`、`no features vs features`、`model vs model + ACI`。
- `feature_ablation/`：有無 causal features 的消融比較圖與表。

目前報告主要採三組評估：

1. `3-day moving average naive baseline vs model`：因模型 `window_size=3`，此 baseline 同樣只使用前三個交易日資料，並以 MAPE / MAE / RMSE 與 MDM 檢定比較點預測誤差。
2. `no features vs features`：固定 decomposition、delta target 與 neural architecture，只比較是否加入 OHLC / technical causal features。
3. `model vs model + ACI`：ACI 不改變點預測，因此不用 MAE / RMSE 比較，而是以 coverage、miss rate、bound-band coverage 與 bound offset 衡量風險邊界效果。

## 重要參數

主要參數集中在 `configs/sp500.yaml`：

- `experiment.window_size: 3`：論文設定的時間視窗。
- `experiment.target_transform: "delta"`：預設讓 decomposition 和 component model 學習 one-step price change，最後用 `PredictedPrice_t = NaivePreviousValue_t + PredictedDelta_t` 還原價格；改成 `"level"` 可回到原本直接預測價格 level 的流程。
- `experiment.train_ratio: 0.8`：前 80% 訓練、後 20% 測試。
- `experiment.train_split_date`：指定日期切分訓練/測試；留空時使用 `train_ratio`。
- `experiment.time_scale: 1`：`1` 是 daily；改成 `5` 可跑 weekly robustness，改成 `20` 可跑 monthly robustness。
- `decomposition.mode: iceemdan_pso_vmd`：proposed model；可改 `iceemdan` 或 `none` 做消融。
- `decomposition.scope: walk_forward`：較嚴格的 no-leakage 預測流程；若要對表復刻論文，可改成 `full_sample`，若要跑較快 baseline 可改成 `train_only_recursive`。
- `features.enabled: true`：將 OHLC 衍生的相對/比例特徵與技術指標接到 VIMF/IMF 模型的 rolling window，不直接餵 raw Open/High/Low/Close 價格 level。這些特徵不參與 ICEEMDAN/VMD 分解，也不會被折回 `Res`；多出的 IMF 仍只會把 IMF 數值折回 `Res`。
- `features.use_for_res: false`：`Res` 模型預設不使用外生特徵，只看自己的 residual window。若要測試 residual 也吃 features，可改成 `true`。
- `decomposition.use_paper_vmd_params: true`：S&P 500 High/Low 使用論文 Table 4 的 `K=3`、`alpha=2455/2740`。若要重新跑 PSO，改成 `false` 或 CLI 加 `--search-vmd`。
- `model.variant: proposed`：可改成 `no_attention`、`no_tcn`、`no_bilstm`、`lstm_sam_tcn`、`bilstm_sam_cnn`。
- `model.dropout: 0.2`
- `model.tcn_channels: 32`
- `model.tcn_kernel_size: 2`
- `model.tcn_dilations: [1, 2, 4, 8]`
- `model.patience: 20`
- `model.search_hyperparameters: false`
- `model.use_cached_params: true`：不開 grid search 時，若有 `best_params_<target>.json` 就直接重用。
- `model.save_best_params: true`：每次訓練後存出各 component 最佳參數。
- `model.retrain_model: true`：重新訓練每個 component 模型，並把 PyTorch 權重、target scaler、feature scaler 存到 `model.checkpoint_dir`。
- `model.checkpoint_dir: "outputs/sp500/model_weights_delta"`：模型權重資料夾。程式會依 target 自動分成 `high/`、`low/`，每個 component 會存成一個 `.pt`。checkpoint 會記錄 `target_transform`，避免誤載舊版 level 模型權重。
- `conformal.enabled: true`：在點預測外加上 Adaptive Conformal Inference 區間。
- `conformal.target_coverage: 0.95`：單側目標 coverage；HighBound 控制 `ActualHigh <= HighBound`，LowBound 控制 `ActualLow >= LowBound`，等價於初始 `alpha=0.05`。
- `conformal.rolling_window: 252`：波動率用過去約一個交易年的 target 日變動標準差估計。
- `conformal.calibration_window: 63`：`q_high` / `q_low` 只用最近約一季的 signed standardized residual 估計，避免舊極端行情讓後續邊界長期偏寬。若要做敏感度，可同時報告 `63`、`126`、`252` 三組。
- `conformal.gamma: 0.005`：ACI 每天根據是否 miss 來微調 alpha；越大反應越快、區間也越容易震盪。

ACI 採用 Gibbs & Candès (2021, NeurIPS) 的適應性共形推論概念；本專案不建立同一 target 的 two-sided prediction interval，而是分別建立 High 的 one-sided upper bound 與 Low 的 one-sided lower bound。本專案不使用 ATR 來放大/縮小邊界，而是用 target 自身的 causal rolling volatility，避免過度依賴金融技術指標語言。

若要更貼近論文的子序列逐一試參數，可把 `model.search_hyperparameters` 改成 `true`，程式會在 `hidden_grid`、`epoch_grid`、`batch_grid` 裡搜尋。

如果已經訓練過，想直接使用儲存的模型權重，把 `model.retrain_model` 改成 `false`，或用 CLI：

```bash
python -m sp500_forecast.cli --config configs/sp500.yaml run --target high --load-weights
```
