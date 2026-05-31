# Evaluation Comparison Tables

Data source notes: with-feature model predictions are taken from the current committed feature run; no-feature predictions are taken from the current outputs/sp500 run. The 3-day naive baseline is causal and uses only t-1 through t-3 actual target values.

## 1. Naive 3-day Average vs Model

| Target | Baseline | Baseline_MAE | Model_MAE | MAE_Improvement_% | Baseline_RMSE | Model_RMSE | RMSE_Improvement_% | Baseline_MAPE_% | Model_MAPE_% | MAPE_Improvement_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High | 3-day moving average | 39.2751 | 32.5413 | 17.1451 | 52.2763 | 42.6494 | 18.4154 | 0.9524 | 0.7924 | 16.8082 |
| Low | 3-day moving average | 44.9980 | 35.9619 | 20.0811 | 58.7619 | 46.9320 | 20.1319 | 1.1067 | 0.8865 | 19.8947 |


## 2. No Features vs Features

| Target | NoFeature_MAE | Feature_MAE | MAE_Improvement_% | NoFeature_RMSE | Feature_RMSE | RMSE_Improvement_% | NoFeature_MAPE_% | Feature_MAPE_% | MAPE_Improvement_% | NoFeature_ACI_Coverage_% | Feature_ACI_Coverage_% | NoFeature_MeanBoundOffset | Feature_MeanBoundOffset | BoundOffset_Reduction_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High | 38.5278 | 32.5413 | 15.5381 | 50.1956 | 42.6494 | 15.0337 | 0.9360 | 0.7924 | 15.3439 | 95.8333 | 96.1667 | 99.4634 | 72.7735 | 26.8339 |
| Low | 44.9348 | 35.9619 | 19.9687 | 58.4058 | 46.9320 | 19.6449 | 1.1093 | 0.8865 | 20.0820 | 96.0000 | 95.8333 | 110.0104 | 95.5754 | 13.1215 |


## 3. Model vs Model + ACI

| Target | PointPrediction_MAE | PointAsBound_Coverage_% | ModelPlusACI_Coverage_% | Coverage_Gain_pp | PointAsBound_MissRate_% | ModelPlusACI_MissRate_% | MeanBoundOffset | MedianBoundOffset | PointPrediction_Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High | 32.5413 | 54.0000 | 96.1667 | 42.1667 | 46.0000 | 3.8333 | 72.7735 | 70.8886 | ACI does not change point prediction |
| Low | 35.9619 | 46.8333 | 95.8333 | 49.0000 | 53.1667 | 4.1667 | 95.5754 | 94.1570 | ACI does not change point prediction |

