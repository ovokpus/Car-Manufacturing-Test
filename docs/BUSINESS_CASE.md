# Business case: Car Manufacturing Test (Mercedes-Benz Greener Manufacturing)

**Predicting how long each vehicle configuration spends on the test bench, so plants can plan bench capacity.**

## Who buys this

The buyer is the manufacturing-engineering or production-planning manager at a plant that builds many configurations of a product and tests each unit before it ships. The obvious example is car assembly, but the same applies to trucks, industrial equipment and appliances. The users are the planners who schedule bench time and the engineers who look for the options that slow testing down.

## The problem and what it costs

When every unit is a different combination of options, test time varies, and planners can't sequence the line or size bench capacity without an estimate. Daimler framed exactly this problem in its 2017 Kaggle competition. Competitors had to predict test-bench time from anonymised configuration features, and the stated goal was "speedier testing, resulting in lower carbon dioxide emissions without reducing Daimler's standards". The competition drew 3,823 teams ([Kaggle, Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing)). The data has hundreds of binary option flags and a handful of categorical ones, with the target `y` in seconds.

I found no public figure for what a bench-second costs, so the ROI below uses assumptions.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Handle hundreds of option flags | The exploration notebook, `Mercedes Benz Greener.ipynb`, removes the outlier and the zero-variance columns, then applies dimensionality reduction (PCA) |
| Predict test time | An XGBoost regressor, packaged as a Python package with a training command |
| Give planners a number before the bench | A FastAPI service (`/docs`, `/health`) and a Streamlit form that scores a configuration |
| Run it anywhere | Docker Compose for the API and form, a Makefile and tests |

## Use-case scenarios

All of these organisations are fictional.

1. **An assembly plant's daily test schedule.** Tomorrow's build list is scored before the shift. Long-test configurations are spread across benches instead of queuing on one, which reduces idle time and overtime.
2. **A configuration engineer's investigation.** Engineers look at which option groups drive the predicted time and pick a candidate for a shorter test procedure, which they then validate on the bench.
3. **A capacity plan for a new model year.** Planners score the expected option mix to estimate how many bench hours the new mix needs before they commit to extra equipment.

## ROI: a worked model (illustrative)

Every figure here is an **illustrative assumption**, because the source data is anonymised and has no cost information.

| Input | Value | Basis |
|---|---|---|
| Vehicles tested per year on one line | 100,000 | Assumption |
| Average bench time per vehicle | 100 s | Assumption, close to the scale of `y` in the competition data |
| Bench time recovered through better sequencing (less idle time between units of different lengths) | 3% | Assumption, to be measured |
| Fully loaded cost of one bench-hour (energy, staff, equipment) | US$300 | Assumption |

Recovered time: 100,000 × 100 s × 3% = 300,000 s, about **83 bench-hours a year, roughly US$25,000** per line. That is modest. The case gets stronger when the plant is short of bench capacity, because then the alternative is buying another bench.

## Path to production

1. Train on the plant's own option codes and measured test times, with named features instead of anonymised ones.
2. Feed the score into the scheduling system, and compare scheduled with actual bench time to measure the real gain.
3. Retrain each model year, when the option mix changes.

`PRODUCTIONIZING.md` lists the engineering steps.

## Risks and limits

- The competition's best scores explained only part of the variance (it was scored on R²), so the estimates are useful for planning, not for promising exact times.
- The data is anonymised and from 2017. Real value needs plant data.
- **The case is moderate.** The problem is real and well documented, but the size of the saving depends entirely on figures only a plant has.

## Sources

- Kaggle, Mercedes-Benz Greener Manufacturing (Daimler, 2017): https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing
