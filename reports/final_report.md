# Titanic Survival Prediction — Final Report

## Problem

Predict whether each passenger in `test.csv` survived the Titanic disaster.
Target `Survived ∈ {0, 1}`; competition metric **accuracy**. All modelling
decisions were validated with stratified cross-validation on the 891 labelled
training rows, never with the hidden test labels or the public leaderboard.

## Dataset

- Train: **891** rows · Test: **418** rows · Binary classification.
- Train target: **342** survived (**38.38%**), **549** died (61.62%).
  Majority-class baseline accuracy is therefore **0.6162**.
- Columns: `PassengerId, Survived (train only), Pclass, Name, Sex, Age, SibSp,
  Parch, Ticket, Fare, Cabin, Embarked`. Train/test schemas verified identical
  modulo the target.

## Data quality

- **Missing**: `Cabin` 77.10% train / 78.23% test → never impute raw values;
  derive `CabinDeck` + `CabinKnown`. `Age` 19.87% / 20.57% → median imputation
  in-pipeline. `Embarked` 2 rows (0.22%), both 1st-class survivors with fare 80
  and cabin B28 → most-frequent imputation. `Fare` 1 missing in test
  (`PassengerId=1044`, 3rd-class Mr. Storey) → median imputation.
- **Duplicates**: 0 exact duplicates in either file — nothing deleted.
- **Types/roles**: `PassengerId` identifier (excluded from modelling);
  `Name`/`Ticket`/`Cabin` high-cardinality text (used only via extracted
  features); `Sex/Embarked/Pclass/Title/CabinDeck/FamilySizeCategory`
  categorical; the rest numeric or binary flags.
- **Key EDA rates** (train): female **74.20%** vs male **18.89%** survival;
  Pclass 1 **62.96%**, 2 **47.28%**, 3 **24.24%**; Embarked C **55.36%**,
  Q **38.96%**, S **33.70%**; fare median 26.0 (survived) vs 10.5 (died);
  age medians both 28.0 (overlap, with a child-survival bump in the KDE);
  `Title`: Mrs **79.37%**, Miss **70.27%**, Master **57.50%**, Rare **34.78%**,
  Mr **15.67%**; family: Small **57.88%**, Alone **30.35%**, Large **16.13%**;
  known cabin **66.67%** vs unknown **29.99%** (confounded with class);
  Sex×Pclass extremes: 1st-class female **96.81%** vs 3rd-class male **13.54%**.

## Feature engineering

`src/feature_engineering.py` (`TitanicFeatureEngineer`, sklearn-compatible so
ticket counts are fit on training folds only):

- `Title` (Mr/Mrs/Miss/Master/Rare, with Mlle→Miss, Mme→Mrs), `FamilySize`,
  `IsAlone`, `FamilySizeCategory` (Alone/Small/Large), `TicketGroupSize`,
  `FarePerPerson = Fare / TicketGroupSize`, `CabinDeck`, `CabinKnown`, `Mother`
  (adult female, `Parch > 0`, Mrs/Miss — documented heuristic).
- `Surname` and `AgeGroup` are EDA-only. `PassengerId` never modelled
  (permutation importance 0.0, as required).
- No target statistics are used; every feature exists at prediction time.

## Models

Six full pipelines (engineering + preprocessing + estimator), 5-fold
stratified CV (`shuffle=True, random_state=42`):

| Model | CV Accuracy | CV Std |
| ----- | ----------: | -----: |
| Gradient Boosting | 0.8406 | 0.0139 |
| Logistic Regression (scaled) | 0.8283 | 0.0158 |
| Hist Gradient Boosting | 0.8260 | 0.0245 |
| Random Forest (300) | 0.8092 | 0.0230 |
| Decision Tree | 0.7901 | 0.0305 |
| Majority baseline | 0.6162 | 0.0023 |

Grid search over small, defensible spaces: RF **0.8440**
(`n_estimators=500, max_depth=12, min_samples_split=5, min_samples_leaf=1,
max_features=None`); GB 0.8429; HGB 0.8417. Full grids and scores are in
`reports/experiment_results.json` and `notebooks/03_model_comparison.ipynb`.

## Validation

- Mean ± std reported across all 5 folds; no single-fold cherry-picking, no
  test-set peeking, preprocessing refit per fold, unseen categories ignored
  safely by the one-hot encoder.
- Ablation (RF, same CV): raw baseline **0.8148 ± 0.0243** vs engineered
  **0.8092 ± 0.0230** vs engineered + tuned **0.8440 ± 0.0141**. The honest
  reading: engineering alone did not measurably exceed the baseline beyond noise — the
  reliable gain required tuning. This is reported as measured, not hidden.
- Experiment log: E1 baseline/LR → E2 tree ensembles → E3 advanced boosting →
  E4 tuned RF/GB/HGB → E5 final refit + submission (see notebooks).

## Best model

**Tuned Random Forest** (params above). Rationale: highest CV mean (0.8440),
lowest re-evaluated std (±0.0141), no extra dependencies, directly
interpretable importances. The 0.0011 margin over tuned GB is noise, so the
tie-break was stability + simplicity — explicitly not "highest single fold".

## Errors

Out-of-fold on train: accuracy **0.8440**, precision **0.8202**, recall
**0.7602**, F1 **0.7891**; TN 492, FP 57, FN 82, TP 260. The model under-detects
survivors (82 false negatives). Error slices: female 18.2% > male 14.2%
(3rd-class/Rare-title females break the prior); Pclass 3/1 ≈ 17–18% vs Pclass 2
8.7%; Rare titles 26.1% (heterogeneous bucket); known-cabin 18.6% vs unknown
14.7% with decks D/A/C worst (small, class-confounded slices); small families
17.5%, alone 15.5%, large 8.1% (near-uniform deaths, easy). Full slice tables:
`data/processed/oof_error_analysis.csv`.

## Results

- Best CV accuracy **0.8440** (std ≈ 0.014); OOF accuracy identical at 0.8440.
- `submissions/submission.csv`: 418 rows, contract-validated
  (`PassengerId` order preserved, binary, complete), 155 predicted survivors
  (**37.1%**, consistent with the 38.4% train prior).
- Kaggle public score: **not claimed** — the file has not been uploaded in this
  session; upload it to record the official number.

## Conclusion

A disciplined, leakage-free workflow reaches ~0.84 CV accuracy with a tuned
Random Forest whose drivers (`Title`, `Fare`/`FarePerPerson`, `Age`, `Pclass`,
`Sex`) match century-old historical understanding of the disaster. The main
lesson is methodological: feature engineering only paid off jointly with
regularised tuning, and the residual errors concentrate exactly where the data
is thinnest (rare titles, small deck slices, class-breaking female survivors).
Next steps worth taking: joint surname+ticket family keys, leakage-safe target
encoding for `Title`, a calibrated RF+GB+HGB stack, and SHAP explanations for
the identified error slices.

Reproduce: `pip install -r requirements.txt`, run `notebooks/01–04` in order
(seed 42, Python 3.14.0, sklearn 1.9.0). Artefacts: `reports/figures/`,
`reports/experiment_results.json`, `models/final_pipeline.joblib`.
