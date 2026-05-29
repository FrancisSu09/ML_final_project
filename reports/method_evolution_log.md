# 方法演化紀錄：從論文復刻到目前版本

這份紀錄整理的是「每一次方法改動背後的原因與邏輯」，不是單純的程式修改清單。整個專案最核心的脈絡是：一開始想參考並復刻 Gong & Xing (2024) 的 S&P 500 High/Low 預測流程，但在實作與檢查過程中發現原始流程容易產生未來資料洩漏，因此逐步把方法改成比較嚴格、可解釋、可防守的 no-leakage 版本，最後再加上外生特徵、ACI 區間與模型權重管理。

## 0. 原始研究目標：參考 Gong & Xing (2024)

最一開始的目標是參考並實作這篇研究：

- Gong & Xing (2024), "Predicting the highest and lowest stock price indices: A combined BiLSTM-SAM-TCN deep learning model based on re-decomposition"
- 原始想法是針對 S&P 500 的 High 與 Low 建立預測模型。
- 論文主流程大致是：
  1. 對原始 High/Low 序列做 ICEEMDAN 分解。
  2. 對第一個高頻 IMF 再做 PSO-VMD re-decomposition。
  3. 得到多個 component：例如 VIMF1、VIMF2、VIMF3、IMF2、IMF3、...、Res。
  4. 每個 component 各自訓練一個 BiLSTM-SAM-TCN 模型。
  5. 把所有 component 的預測值加總，得到最終 High 或 Low 的點預測。

原始復刻的直覺是合理的：金融價格序列同時有短期震盪、中期波動、長期趨勢；先把訊號拆成不同頻率的 component，再分別預測，理論上比直接預測原始價格更容易。

但後來最大的問題不是模型架構，而是「分解這一步到底有沒有偷看到未來」。

## 1. 第一版：full-sample decomposition，接近論文復刻但有 leakage 風險

第一版比較接近論文式的做法：先對完整序列做分解，再把分解後的 component 切成訓練與測試。

這樣做的好處是：

- 最接近原論文流程，容易復刻表格與圖。
- 每個 component 的長度完整，component 數量也固定。
- 訓練、驗證、測試都可以直接從同一組分解結果拿資料，實作簡單。

但是後來發現這樣有很嚴重的資料洩漏問題。

原因是 ICEEMDAN / VMD 這類 decomposition 不是逐日獨立的轉換，而是會用整段序列的形狀來決定 IMF / VIMF / Res。如果先把 2011-2023 全部資料一起分解，再回頭切 train/test，那測試期間的價格形狀其實已經影響了訓練期間的 component 表示。

也就是說，就算模型訓練時只拿 train 區間，train 區間的 IMF 本身也已經被 test 區間影響過。這不是時間序列預測時真正能取得的資訊。

所以這版可以保留作為「paper-style replication」或論文對照，但不能當成最嚴謹的 no-leakage 結果。現在程式仍保留這個模式：

```yaml
decomposition:
  scope: "full_sample"
```

它的定位是復刻用，不是主結果。

## 2. 第二版：fit-only decomposition / train_only_recursive，先嘗試完全避開 leakage

發現 full-sample decomposition 有 leakage 後，下一步的想法是：那就只對訓練區間做 decomposition。

這個版本的核心邏輯是：

- 只拿 fit / train 區間做 ICEEMDAN-PSO-VMD。
- 用 fit 區間分解出來的 component 訓練 component models。
- 驗證與測試期間不要重新對包含未來的完整序列做 decomposition。
- 未來 component 只能由模型自己遞迴預測出去。

這樣的優點是很乾淨：fit 階段完全沒有看到 validation/test 的價格資料，因此可以解決第一版的 leakage 問題。

但實際跑起來後，它產生另一個問題：遞迴誤差累積太嚴重。

因為模型不是只預測明天，而是要在 component space 裡一路把未來 component 推出去。第 1 天 component 預測錯一點，第 2 天的輸入就已經包含錯誤，第 3 天又建立在前面的錯誤上，越往後越容易漂移。

尤其是測試區間很長時，這種 recursive forecasting 會讓 component sequence 慢慢偏離真實序列，最後加總後的 High/Low 也會跟實際價格脫節。

所以這個版本雖然 no-leakage，但結果不穩。現在程式仍保留它作為 baseline：

```yaml
decomposition:
  scope: "train_only_recursive"
```

它的定位是「最嚴格但容易漂移的 baseline」，不是目前主流程。

## 3. 第三版：方案 B，改成 walk-forward decomposition

為了同時兼顧 no-leakage 與預測穩定性，後來改成方案 B：walk-forward decomposition。

這是目前專案的主流程：

```yaml
decomposition:
  scope: "walk_forward"
```

它的核心想法是：預測第 t 天時，只能使用第 t 天以前已經知道的歷史資料。

具體流程如下：

1. fit 階段只用訓練資料做 decomposition，訓練 component models。
2. 到 validation/test 的某一天 t 時，不使用第 t 天或之後的價格。
3. 對 `0 ... t-1` 的歷史序列重新 decomposition。
4. 從這次 decomposition 的最後幾天 component window 取出輸入。
5. 用 fit 階段訓練好的 component model 預測第 t 天。
6. 每一天都重複這個流程，形成 walk-forward 預測。

這樣做的好處是：

- 沒有使用未來資料，因為第 t 天的輸入只來自 `t-1` 以前。
- 不需要讓 component model 自己遞迴預測很長一段未來，因此比 train_only_recursive 穩定。
- 更接近真實交易或真實預測情境：每天只知道昨天以前的資料。

目前程式中的 `_precompute_walk_forward_windows(...)` 就是這個邏輯：每個 target day 都用 `signal[:target_index]` 重新分解，也就是只用到前一天為止的歷史。

這一步是整個專案方法論上最重要的轉折：從「接近論文但可能 leakage」轉成「比較嚴格的 no-leakage walk-forward」。

## 4. Walk-forward 的新問題：每天重新分解時 component 數量可能不一致

改成 walk-forward 後，又遇到新的實務問題：每天重新 decomposition 時，產生的 IMF 數量不一定和 fit 階段相同。

例如 fit 階段訓練時有：

```text
VIMF1, VIMF2, VIMF3, IMF2, IMF3, IMF4, IMF5, IMF6, IMF7, Res
```

但某一天 walk-forward decomposition 可能多出：

```text
IMF8, IMF9
```

這會造成模型對不上，因為 fit 階段沒有訓練過 IMF8 / IMF9 的模型。

當時可選的解法有幾個：

1. 直接丟掉多出來的 IMF。
2. 動態新增模型去預測多出來的 IMF。
3. 把多出來的 IMF 折回 `Res`。

最後選擇第 3 種：把 walk-forward 時多出 fit 階段沒有的 IMF 折回 `Res`。

理由是：

- 直接丟掉多出來的 IMF 會破壞原訊號的加總關係，等於少了一部分訊號能量。
- 動態新增模型會讓每一天的模型結構不一致，方法很難解釋，也很難和 fit 階段對齊。
- `Res` 本來就是 residual / remainder / trend bucket，把未被 fit-time component layout 接住的額外 IMF 放回 `Res`，比較符合 decomposition 的加法結構。

所以目前策略是：

- 如果 walk-forward decomposition 有 fit 階段沒有的額外 component，優先 fold into `Res`。
- 如果 component list 裡沒有 `Res`，才退而求其次 fold into 最後一個 component。

程式邏輯在：

- `_align_walk_forward_components(...)`
- `_extra_component_fold_target(...)`

也有測試確保 IMF8 / IMF9 會折回 Res：

- `test_walk_forward_alignment_folds_extra_components_into_residual`

這一版的重點是：walk-forward 解決 leakage，但會帶來 component 對齊問題；fold-extra-into-Res 是為了維持固定模型結構與保留訊號總量。

## 5. 模型架構：BiLSTM -> SAM attention -> TCN -> dense head 的位置

每一個 component 都會有一個自己的神經網路模型。以目前主模型 `proposed` 來說，單一 component model 的流程是：

```text
component window + optional feature window
        |
        v
BiLSTM
        |
        v
SAM / additive self-attention
        |
        v
TCN
        |
        v
dense head
        |
        v
next-day component prediction
```

也就是說，BiLSTM -> SAM attention -> TCN -> dense head 不是在 decomposition 之前，而是在每個分解後的 component model 裡面。

整體關係可以理解成：

```text
Raw High/Low series
        |
        v
ICEEMDAN + PSO-VMD decomposition
        |
        v
VIMF/IMF/Res components
        |
        v
one BiLSTM-SAM-TCN per component
        |
        v
sum component predictions
        |
        v
final High/Low prediction
```

也就是「先分解，再對每個 component 各自建模，最後加總」。

## 6. Feature 加入模型：怎麼篩、怎麼餵進去

原始論文主要是對分解後的 component 做 deep learning 預測；後來我們加上外生特徵，是因為只看 component 本身可能會少用掉 OHLC 的上下文資訊。

但加入 feature 時有兩個原則：

1. 不能用 target day 的資料。
2. 不能直接把原始價格 level 亂塞進去，避免尺度與趨勢干擾。

所以目前 feature 設計是：

- 使用 OHLC 衍生的比例特徵，例如相對前收盤價的漲跌幅。
- 使用技術型特徵，例如 SMA ratio、return std、RSI、MACD、RollingVolatility20、Bollinger bandwidth。
- feature window 和 component window 一樣使用過去 `window_size` 天。
- 預測第 t 天時，只用 `t-window_size ... t-1` 的 features，不用第 t 天的 features。

目前 High 輸出摘要中的 feature 數量是 19 個：

```text
CloseReturn
LogCloseReturn
HighPrevClosePct
LowPrevClosePct
HighLowRangePct
OpenPrevClosePct
CloseOpenChangePct
CloseSMA5Ratio
ReturnStd5
CloseSMA10Ratio
ReturnStd10
CloseSMA20Ratio
ReturnStd20
RSI14
MACDPct
MACDSignalPct
MACDHistPct
RollingVolatility20
BollingerBandwidth20
```

因此目前輸入維度是：

- VIMF / IMF component：每個 timestep 有 `1 個 component 值 + 19 個 features = 20 維`。
- Res component：每個 timestep 只有 `1 個 Res 值`，因為目前設定是：

```yaml
features:
  use_for_res: false
```

Res 不吃外生 feature 的理由是：Res 通常代表比較慢的趨勢或殘差主體，如果把短期技術特徵也塞進去，容易把主趨勢拉得不穩。現在先讓 Res 維持 component-only，VIMF/IMF 才接 features。

## 7. window_size = 3 的來源

目前設定：

```yaml
experiment:
  window_size: 3
```

這個 3 不是 walk-forward 自己產生的，也不是 ACI 的參數，而是沿用論文模型設定中的時間視窗概念：用前 3 個 timestep 預測下一個 timestep。

在程式裡，`window_size=3` 的意思是：

```text
用第 t-3, t-2, t-1 天的 component / feature window
預測第 t 天的 component
```

如果 time scale 是日資料 `T=1`，那就是用最近 3 個交易日預測下一個交易日。

目前保留 `3` 的原因是：

- 它對應論文設定，方便和原研究對照。
- 目前樣本數約 3000 筆，window 太長會減少可訓練樣本，也會加重每個 component model 的複雜度。
- 後續如果要做敏感度分析，可以比較 `3, 5, 10, 20` 等不同 window，但目前主結果先保留論文設定。

## 8. ATR 改成 RollingVolatility20：讓解釋更直觀

一開始曾經使用 ATR 類型的波動特徵，但後來決定拿掉 ATR，改成 `RollingVolatility20`。

原因是 ATR 比較偏金融技術分析術語，對老師或非金融背景讀者不一定直觀。相比之下，RollingVolatility20 比較容易解釋：

```text
RollingVolatility20 = 過去 20 天 close return 的 rolling standard deviation
```

也就是「最近一段時間報酬率波動有多大」。

這個改法有兩個好處：

- 解釋上比 ATR 簡單。
- 後面 ACI 使用 `Error_t / Volatility_t` 做標準化時，RollingVolatility 的概念可以自然銜接。

目前 source code 已經使用 `RollingVolatility20`，不再產生 `ATRPct`。如果舊的 Low summary 裡還看到 `ATRPct`，代表那是改名前的舊輸出，需要重新跑 Low 才會刷新。

## 9. 加入 ACI：從點預測變成有 coverage 概念的邊界

在有 High / Low 點預測後，下一個問題是：只給一條 prediction line 不夠，因為老師可能會問這個預測有沒有不確定性範圍。

所以後來加上 ACI，也就是 Gibbs & Candes (2021) 提出的 Adaptive Conformal Inference。

這裡的直覺是：

- 模型先給點預測 `Predicted_t`。
- 再根據近期預測誤差估計一個安全邊界。
- 如果市場波動大，邊界自然放寬。
- 如果市場波動小，邊界自然收窄。
- ACI 會根據最近 coverage 狀況動態調整 alpha。

目前改成 one-sided signed ACI，不再用 `|error|` 把方向拿掉。先定義 signed standardized residual：

```text
r_t = (Actual_t - Predicted_t) / Volatility_t
```

這裡 `r_t > 0` 代表模型低估，`r_t < 0` 代表模型高估。High bound 使用近期 `r_t` 的上尾分位數：

```text
HighBound_t = PredictedHigh_t + q_high,t * Volatility_t
ErrHigh_t = 1{ActualHigh_t > HighBound_t}
```

Low bound 使用近期 `r_t` 的下尾分位數：

```text
LowBound_t = PredictedLow_t + q_low,t * Volatility_t
ErrLow_t = 1{ActualLow_t < LowBound_t}
```

這一步回應了兩個重點：第一，不能直接拿原始價格誤差去算 q，否則 q 可能被極端行情拉走；第二，不能只用絕對值誤差，否則會丟掉「模型偏高或偏低」的方向。

因此目前用的是 signed standardized residual：

```text
r_t = (Actual_t - Predicted_t) / Volatility_t
```

然後 High 對近期 `r_t` 取上尾分位數得到 `q_high,t`，Low 對近期 `r_t` 取下尾分位數得到 `q_low,t`。最後再乘回當天的 `Volatility_t`，得到符合當天波動狀態與模型偏差方向的邊界。

這樣比較好解釋：

- `q_high` 代表「Actual High 超過 Predicted High 時，通常會超過幾個 volatility」。
- `q_low` 代表「Actual Low 低於 Predicted Low 時，通常會低幾個 volatility」。
- Volatility_t 代表「今天市場本身有多不穩」。
- q 與 Volatility 相乘後，才是今天應該調整的價格邊界。

## 10. ACI 的 rolling q：避免舊極端行情長期污染邊界

一開始如果用全部歷史 signed standardized residual 估計 q，會有一個問題：很久以前的極端行情會一直留在 calibration set 裡，導致後面行情平穩時，邊界仍然過寬。

所以後來改成 `q_high` / `q_low` 只使用最近一段 calibration window 的 signed standardized residual。

目前設定是：

```yaml
conformal:
  rolling_window: 252
  calibration_window: 63
```

兩個 window 的意義不同：

- `rolling_window: 252` 是用來估計 volatility，約等於一年交易日。
- `calibration_window: 63` 是用來估計 `q_high` / `q_low`，約等於一季交易日。

選 63 的邏輯是：

- 比 20 天穩，不會因為少數幾天誤差大就劇烈跳動。
- 比 252 天反應快，不會被很久以前的極端行情拖住。
- 一季交易日也比較容易在報告中說明。

後續可以做敏感度分析：`63 / 126 / 252`，但目前主結果先用 63 作為「近期 calibration」。

## 11. Plot 改成只畫目標方向的 ACI band

一開始圖上可能會畫完整上下區間，但對 High 預測來說，最重要的是上界；對 Low 預測來說，最重要的是下界。

所以圖的呈現改成：

- High plot：畫 `Predicted` 到 `HighBound` 的上方區間。
- Low plot：畫 `LowBound` 到 `Predicted` 的下方區間。

這樣圖更貼近任務：

- High 任務關心「實際最高價是否落在上界以內」。
- Low 任務關心「實際最低價是否落在下界以內」。

因此現在 legend 中會看到 `ACI upper band` 或對應的 lower band。

## 12. 模型權重存取：加入 retrain/load switch

後來又加上模型權重儲存與讀取，是因為每次完整訓練所有 component model 都很耗時，而且結果不容易重現。

目前設定：

```yaml
model:
  retrain_model: true
  checkpoint_dir: "outputs/sp500/model_weights"
```

邏輯是：

- `retrain_model: true`：重新訓練每個 component model，並把權重存到 checkpoint directory。
- `retrain_model: false`：不重新訓練，直接從 checkpoint directory 載入已儲存的模型權重。

儲存的內容不只有 PyTorch model weights，還包含：

- model architecture metadata
- target scaler
- feature scaler
- component name
- training params

這樣做是必要的，因為如果只存 neural network weights，沒有 scaler 或 architecture 設定，之後載入時很容易輸入尺度或模型結構對不上。

這個功能讓後續可以：

- 先花時間訓練一次。
- 之後用相同權重重畫圖、重算 ACI、改 summary，不需要每次重訓。
- 比較不同後處理策略時更公平，因為點預測模型固定。

## 13. 目前主流程總結

目前最推薦、最能解釋的主流程是：

```text
Raw S&P 500 High / Low
        |
        v
One-step price change target
        |
        v
Train / validation / test split
        |
        v
Fit period only decomposition
        |
        v
ICEEMDAN + PSO-VMD components
        |
        v
For each component:
  component window
  + exogenous features for VIMF/IMF
  + no features for Res by default
        |
        v
BiLSTM -> SAM attention -> TCN -> dense head
        |
        v
component next-day prediction
        |
        v
sum component change predictions
        |
        v
PredictedDelta
        |
        v
PredictedPrice_t = NaivePreviousValue_t + PredictedDelta_t
        |
        v
one-sided signed ACI:
(actual - predicted) / volatility
        |
        v
HighBound / LowBound
        |
        v
BoundBandCovered and bound_band_coverage
```

## 14. 各版本的定位

目前可以把各版本定位成這樣：

| 版本 | 解決的問題 | 主要缺點 | 現在定位 |
| --- | --- | --- | --- |
| full_sample | 接近論文復刻，流程簡單 | 有未來資料洩漏風險 | 只作 paper-style 對照 |
| train_only_recursive | 完全避免 test 進入 decomposition | 遞迴誤差累積，長期容易漂移 | no-leakage baseline |
| walk_forward | 避免 leakage，又減少長期遞迴漂移 | 每天分解成本高，component 數可能不一致 | 目前主流程 |
| walk_forward + fold extras into Res | 解決 component mismatch | Res 會承接額外 IMF，需在報告說明 | 目前主流程的一部分 |
| features | 補充 OHLC 與技術上下文 | 需嚴格避免 target-day feature leakage | 目前主流程 |
| delta target | 減少直接價格 level 外推造成的系統性高估/低估 | 圖與 summary 要重新解釋成「預測變化量後還原價格」 | 目前主流程 |
| ACI | 給出有 coverage 概念的邊界 | 不改善點預測本身，只校準區間 | 目前主流程 |
| checkpoints | 避免每次重訓，提高可重現性 | 權重需和 config / scaler 對應 | 工程支援功能 |

## 15. 口頭說明時可以用的版本

如果要跟老師簡短說明整個方法演化，可以說：

> 我們一開始參考 Gong & Xing (2024) 的 ICEEMDAN-PSO-VMD-BiLSTM-SAM-TCN 架構，先把 High/Low 序列分解成多個 component，再對每個 component 建模後加總。但實作後發現，如果像論文復刻那樣先對完整序列分解再切 train/test，測試期間資料會影響訓練期間的 IMF 表示，形成資料洩漏。因此我們先嘗試只對 fit 區間分解並遞迴預測未來 component，雖然避免 leakage，但長期會累積誤差。最後改成 walk-forward：預測第 t 天時，只用 t-1 以前的歷史重新分解，這樣既不看未來，也比純遞迴穩定。walk-forward 又遇到每天分解出的 IMF 數可能不同，因此我們把 fit 階段沒有的額外 IMF 折回 Res，維持固定 component layout 與訊號加總關係。後來我們發現直接用價格 level 訓練時，train/test 價格水準一變就會產生系統性低估或高估，所以保留 decomposition 與 component model 架構，但把 target 改成 one-step price change：先分解變化量、預測下一期變化量，再用前一日價格還原成 High/Low。之後再加入 causal OHLC features、用 RollingVolatility 取代 ATR，並在點預測上加上 one-sided signed ACI，以 `(actual - predicted) / volatility` 保留誤差方向，High 用上尾分位數建立 upper bound，Low 用下尾分位數建立 lower bound，使 HighBound / LowBound 能隨市場波動與模型偏差方向自適應調整。
