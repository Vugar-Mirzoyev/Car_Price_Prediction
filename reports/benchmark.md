## Model Benchmark

All 8 models share one preprocessing pipeline and one train/test split (80/20, `random_state=42`).
Reproduce with `python benchmark.py --data <path> --target sellingprice --drop mmr --cv 5`.

| Model | R² (test) | MAE | RMSE | Fit (s) | 5-fold CV R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9514 | 1,262 | 2,144 | 7.2 | 0.9462 ± 0.0044 |
| Random Forest | 0.9479 | 1,289 | 2,218 | 49.0 | 0.9437 ± 0.0030 |
| LightGBM | 0.9387 | 1,495 | 2,407 | 4.1 | 0.9354 ± 0.0028 |
| HistGradientBoosting | 0.9339 | 1,515 | 2,499 | 8.6 | 0.9288 ± 0.0038 |
| CatBoost | 0.9147 | 1,747 | 2,839 | 8.4 | 0.9130 ± 0.0024 |
| Decision Tree | 0.9101 | 1,796 | 2,914 | 5.3 | 0.9070 ± 0.0030 |
| Linear Regression | 0.7805 | 3,037 | 4,555 | 2.6 | 0.7801 ± 0.0031 |
| AdaBoost | 0.0403 | 8,347 | 9,523 | 33.4 | 0.0124 ± 0.0690 |

**Selected model: XGBoost** - R² 0.9514, MAE 1,262 on the held-out test set.

### Not benchmarked

- **KNN** - prediction is quadratic in dataset size; 447,048 training rows exceeds the 50,000-row budget. Re-run with --max-rows 50000 to include it.

Every tree-based model is depth-bounded so the run is reproducible in minutes rather than hours; the bounds are not tuned, and no model receives one another does not. `HistGradientBoosting` stands in for sklearn's exact `GradientBoostingRegressor`, which is single-threaded and does not finish at this scale.
