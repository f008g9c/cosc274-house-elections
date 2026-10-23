# cosc274-house-elections

Predicting U.S. House election outcomes from historical district-level returns (1976–2024), comparing logistic regression, perceptron, SVM, and k-NN classifiers, with a squared-error vs. LAD comparison for vote-margin regression. COSC 274 final project.

## Project structure

```
data/
  raw/          # MEDSL source file (not committed)
  processed/    # generated modeling table (not committed)
src/
  build_dataset.py   # raw returns -> one row per district-year with features
notebooks/      # exploration and figures
results/        # saved metrics and plots
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

U.S. House 1976–2024, MIT Election Data and Science Lab (MEDSL), Harvard Dataverse.
Download the main data file into `data/raw/`, then run:

```bash
python src/build_dataset.py --input data/raw/<filename>
```

## Experimental design

| Split      | Elections          |
|------------|--------------------|
| Train      | 1978–2014          |
| Validation | 2016               |
| Test       | 2018 (used once)   |
| Stress test| 2022 (post-redistricting) |

## Features (signed from the Democratic point of view)

- `margin_lag1`, `margin_lag2`: two-party margin in the same district one and two cycles earlier
- `incumbent`: +1 Democratic incumbent, −1 Republican incumbent, 0 open seat
- `midterm`, `pres_party`, `midterm_x_pres`: national context and the midterm penalty
- `redistrict_year`: first election under new post-census maps
