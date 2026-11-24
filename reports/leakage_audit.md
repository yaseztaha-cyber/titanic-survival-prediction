# Data Leakage & Validation Integrity — Audit (measured)

All checks below are executed programmatically by `src/analysis.py::leakage_audit`
(see `reports/extended_results.json → leakage`).

## L1 Target never a model input — PASS

Evidence: model inputs=16 cols; 'Survived' in DROP_COLUMNS=True

## L2 PassengerId never a model input — PASS

Evidence: 'PassengerId' in DROP_COLUMNS=True; not in numeric/categorical lists

## L3 No target reference in feature-engineering code — PASS

Evidence: code-level references to 'Survived' in src/feature_engineering.py (docstrings/comments excluded): 0

## L4 Preprocessing inside CV folds (Pipeline order) — PASS

Evidence: steps=['features', 'preprocess', 'model']; OneHotEncoder(handle_unknown='ignore')

## L5 Test set carries no labels — PASS

Evidence: test columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

## L6 PassengerId permutation importance is 0.0 — PASS

Evidence: measured importance_mean=0.0

## L7 Group features use records only, no labels — PASS

Evidence: ticket '347082': count fit-on-all=7 vs fit-on-first-200=2; unseen ticket maps to 1; TitanicFeatureEngineer.fit signature=(self, X: 'pd.DataFrame', y: 'pd.Series | None' = None) -> "'TitanicFeatureEngineer'" (y unused)

## Test-set isolation

The test set (`test.csv`, 418 rows) carries no `Survived` column (check L5). It was used exactly once: 
`best_pipe.predict/predict_proba(test_df)` for the submission and probability artefacts. No feature choice, 
hyperparameter, threshold, or model-selection decision consulted test data.

## CV vs leaderboard

Model selection used 5-fold stratified CV accuracy only (seed 42). Best CV: RF 0.8440 
± 0.0141 vs GB 0.8429 (Δ=0.0011, within noise). 
The Kaggle leaderboard is NOT a validation set: no score has been fetched and no decision depends on it. 
Kaggle status: NOT UPLOADED.

## Group-feature note

TicketGroupSize / FarePerPerson use passenger *records* (ticket strings, fares) — information available at 
prediction time — and never survival labels (checks L3, L7). Counts are fit on training folds only; unseen 
test tickets map to group size 1.