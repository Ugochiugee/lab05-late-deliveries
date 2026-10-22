# Late Deliveries

QuickBite wants to know, when an order is placed, whether it will arrive late so it can send a $4 voucher before the customer is disappointed. This project trains a scikit-learn pipeline that predicts the probability an order is late, and picks the flagging threshold that gives the lowest total cost under the company's voucher rule.

## Setup

```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```
python -m latedeliveries.train
```

This rebuilds `reports/cv_results.csv`, `reports/leakage_check.csv`, `reports/threshold_costs.csv`, `reports/metrics.json`, `models/late_orders.joblib` and `models/model_info.json` from nothing.

## Check it

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns

The model is used when the order is placed, so a column is only a feature if its value is known at that moment.

| Column | Used? | Reason |
|---|---|---|
| distance_km | Yes, numeric | Known from the restaurant and delivery addresses |
| items | Yes, numeric | Known from the basket |
| hour | Yes, numeric | Known from the order time |
| prep_minutes | Yes, numeric | Treated as known at order time (see Known limitations). About 6% missing, filled with the training median |
| courier_months | Yes, numeric | Treated as known once a courier is assigned to the order |
| weather | Yes, category | Known at order time |
| day_type | Yes, category | Known at order time |
| vehicle | Yes, category | Known once a courier is assigned |
| order_id | **No** | A label for the row. It carries no information about lateness, and any pattern a model found in it would be noise |
| customer_rating | **No** | Given by the customer after delivery, so it does not exist when the order is placed. It also depends directly on whether the order was late (late orders average a rating of about 3.0, on-time orders about 4.5). Using it is target leakage |
| late | **No** | The target |

**Leakage check.** I re-ran logistic regression with `customer_rating` added back. Cross-validated ROC-AUC jumped from 0.795 to 0.907 and F1 from 0.352 to 0.650 (`reports/leakage_check.csv`). That jump comes from information the model could never have in production, so the column stays out.

### Split

Stratified 75/25 split with `random_state=42`, held in the constant `SEED` in `pipeline.py`. 2,250 training orders and 750 test orders, both with a late rate of 15.6%. Steps 2 to 4 below use only the training data.

### Model comparison

Five-fold stratified cross-validation on the training data. Every model runs inside the same pipeline (median fill and scaling for numbers, one-hot encoding with `handle_unknown="ignore"` for categories). The last column is the lowest total cost each model reaches on its out-of-fold probabilities across thresholds 0.05 to 0.95. Flagging nobody on the training set costs $5,265.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Lowest CV cost (threshold) |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | $5,265 (0.05) |
| logistic_regression | 0.862 ± 0.009 | 0.654 ± 0.065 | 0.242 ± 0.053 | 0.352 ± 0.064 | 0.795 ± 0.034 | $4,031 (0.30) |
| logistic_regression_balanced | 0.737 ± 0.019 | 0.333 ± 0.027 | 0.681 ± 0.049 | 0.447 ± 0.034 | 0.795 ± 0.034 | $4,098 (0.70) |
| random_forest_balanced | 0.787 ± 0.017 | 0.382 ± 0.034 | 0.584 ± 0.058 | 0.461 ± 0.040 | 0.782 ± 0.025 | $4,237 (0.60) |
| hist_gradient_boosting | 0.851 ± 0.006 | 0.544 ± 0.036 | 0.288 ± 0.037 | 0.375 ± 0.036 | 0.754 ± 0.024 | $4,327 (0.30) |

The dummy baseline gets 84.4% accuracy by never predicting late, which is why accuracy is not used to choose.

**Choice: logistic regression (no class weighting).** The F1 and recall columns are measured at the default 0.5 threshold, but this project does not use 0.5. It picks the threshold from costs. What matters, then, is how well a model ranks late orders above on-time ones (ROC-AUC) and how cheap it is once its threshold is set. Logistic regression has the highest ROC-AUC (0.795) and the lowest out-of-fold cost ($4,031). The balanced version ranks orders almost identically; class weighting mainly shifts its probabilities upward, which is why its best threshold moves to 0.70. Random forest and gradient boosting cost more and have lower ROC-AUC. The differences between the top models are inside one standard deviation, so the simpler, interpretable model is the reasonable pick.

### Tuning

`GridSearchCV` over `model__C` in [0.01, 0.1, 1, 10] and `model__class_weight` in [None, "balanced"], 8 combinations, same five folds. Scoring metric: **ROC-AUC**, because the threshold is chosen separately from costs, so tuning should improve how the model ranks orders, not its score at one fixed cut-off.

Best settings: `C=10`, `class_weight=None`. Best CV ROC-AUC 0.7952 against 0.7951 untuned. The gain (0.0001) is far smaller than the fold-to-fold standard deviation (0.034), so **tuning did not meaningfully beat the untuned model**. The tuned settings are kept because they are no worse, but the result would be the same with the defaults.

### Threshold

A voucher costs $4 for every flagged order, and a missed late order costs $15. The default threshold of 0.5 is wrong for these costs: if the probabilities are accurate, a flag pays for itself once an order has more than a 4/15 ≈ 27% chance of being late. The threshold is chosen from the data instead.

`cross_val_predict` gives out-of-fold probabilities for the 2,250 training orders using the tuned pipeline. For every threshold from 0.05 to 0.95 in steps of 0.05, `cost_by_threshold` in `evaluation.py` flags orders with probability at or above it and adds up the cost. The lowest is **0.30** ($4,035 on the training data, flagging 330 orders). The test set plays no part in this choice.

## Result

The final pipeline was fitted on all 2,250 training orders and scored on the 750 test orders once, at threshold 0.30 (`reports/metrics.json`).

| | Not flagged | Flagged |
|---|---|---|
| Actually on time | 578 | 55 |
| Actually late | 64 | 53 |

Precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796. 108 orders flagged out of 750.

| Policy | Total cost on test set |
|---|---|
| Flag nobody (current) | $1,755 |
| Flag everybody | $3,000 |
| Model at 0.30 | **$1,392** |

The model saves $363 (about 21%) against the current policy. The test set holds only 117 late orders, so this is one estimate from a small sample. I would trust that the model is cheaper than flagging nobody, since the cross-validated saving points the same way (about 23%), but not the exact dollar figure.

## What the tests cover

20 tests in `tests/`, all passing:

- The feature lists leave out `order_id`, `customer_rating` and `late` (parametrized).
- The split is 75/25, stratified, repeatable with the same seed, and has no overlap between train and test.
- Predicted probabilities are between 0 and 1 and each row sums to 1.
- The pipeline predicts for an order with a missing `prep_minutes` and an unseen weather value (`"hail"`).
- `order_cost` on a six-order example worked out by hand, and `cost_by_threshold` at four thresholds on a four-order example (parametrized).
- `choose_threshold` picks flag-nobody (0.95) when that is cheapest, flag-everybody (0.05) when that is cheapest, and a middle threshold on a third example.
- `policy_costs` and the confusion matrix from `threshold_metrics`.
- A pipeline saved with joblib to `tmp_path` gives identical probabilities after reloading, and `model_info.json` matches the feature lists and scikit-learn version in the code.

Not tested: the exact numbers in `reports/metrics.json`. Those depend on the scikit-learn version and the full training run, so a test pinning them would break on a harmless library update. The fresh-clone check in the lab instructions covers that instead.

## Known limitations

- **prep_minutes may not be known at order time.** The data dictionary describes it as the minutes the kitchen took, which is only known once the food is ready. I kept it because the lab requires the pipeline to handle a missing `prep_minutes`, which implies it is a feature. In production it should be replaced by the restaurant's estimated prep time. If the actual value is not available, the model would need retraining without it.
- **courier_months and vehicle** assume a courier is assigned when the order is placed. If assignment happens later, these columns have the same problem.
- **Small test set.** 117 late orders means the test-set cost has a wide margin of error.
- **Probability calibration was not checked.** The threshold is chosen directly from costs, so calibration is not needed for the decision, but it would be needed if the probabilities were shown to staff as percentages.
- **Fixed costs.** The $4 and $15 figures come from the scenario. If they change, the threshold must be chosen again.
