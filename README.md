# cosc274-house-elections

Predicting U.S. House election outcomes from historical district-level returns (1976–2024), comparing standard optimization-based models with Bayesian models whose priors are estimated from earlier elections, for both predicting winners and predicting vote margins. COSC 274 final project.

## Project structure

```
data/
  raw/          # MEDSL source file (not committed)
  processed/    # generated modeling table (not committed)
src/
  build_dataset.py   # raw returns -> one row per district-year with features
notebooks/
  01_explore.ipynb   # exploratory data analysis
  02_models.ipynb    # main workflow: models, evaluation, results (submission notebook)
results/        # saved metrics and figures
```

## Setup

```bash
conda env create -f env.yml
conda activate house-elections
```

In VS Code, select the `house-elections` environment as the notebook kernel. (`requirements.txt` lists the same packages for anyone using pip.)

## Data

U.S. House 1976–2024, MIT Election Data and Science Lab (MEDSL), Harvard Dataverse.
Download the main data file into `data/raw/`, then run:

```bash
python src/build_dataset.py --input data/raw/<filename>
```

## Experimental design

| Split       | Elections                 |
|-------------|---------------------------|
| Train       | 1978–2014 (Bayesian model: 1978–1998 sets the prior, 2000–2014 updates it) |
| Validation  | 2016                      |
| Test        | 2018 (used once)          |
| Stress test | 2022 (post-redistricting) |

## Research question

Does using older elections as a prior produce better-calibrated forecasts than pooling all the data together?

## Models

| Task | Models |
|---|---|
| Classification (who wins) | Logistic regression (cross-entropy), random forest, Naive Bayes with a historical prior, Bayesian logistic regression with a historical prior |
| Regression (vote margin)  | Least squares, least absolute deviations (LAD), Bayesian linear regression with a historical prior |

For every Bayesian model, elections from 1978–1998 set the prior (as pseudo-counts or a prior distribution, scaled by a weight a₀) and 2000–2014 update it. All models see the same 1978–2014 training elections.

## Evaluation

- Baseline: the party that won the district last time wins again (88.2% accurate on 2018)
- Classification: log loss, ROC/AUC, precision/recall/F1, accuracy on competitive races, calibration
- Regression: RMSE, MAE, and coverage of Bayesian credible intervals

## Features (signed from the Democratic point of view)

- `margin_lag1`, `margin_lag2`: two-party margin in the same district one and two cycles earlier
- `incumbent`: +1 Democratic incumbent, −1 Republican incumbent, 0 open seat
- `midterm`, `pres_party`, `midterm_x_pres`: national context and the midterm penalty
- `redistrict_year`: first election under new post-census maps