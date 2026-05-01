# Venture Outcomes Under Censoring

A publication-style exploratory analysis of Crunchbase startup investment data, focused on how funding depth, founding cohort, and market category relate to startup outcomes under strong survivorship and censoring bias.

## Project Summary

This repository turns a public startup investment dataset into a polished analytical report. The central question is not simply "which startups succeed?" but how to interpret startup status when many companies are still operating because they have not yet had enough time to exit or fail.

## Live Report

Published report:

- https://rf2960.github.io/market-investment-visualization/

## Why This Project Is Stronger Now

- Reframes the analysis around censoring, survivorship bias, and time-to-outcome.
- Uses a cleaner publication-style static report instead of a basic class-report landing page.
- Adds custom SVG figures generated from the raw data.
- Separates descriptive evidence from causal claims.
- Includes reproducible report generation through `scripts/build_publication_report.py`.

## Key Findings

- Most companies are marked operating, so direct comparison of status rates is biased by company age.
- In mature 2000-2008 founding cohorts, acquisition and closure rates are more interpretable than in recent cohorts.
- Funding depth is strongly associated with outcomes, but should be read as selection and maturity rather than proof that funding causes success.
- Top startup markets have distinct "exit fingerprints": some show higher acquisition balance, while others show more closure pressure.

## Repository Structure

```text
.
|-- scripts/
|   `-- build_publication_report.py
|-- docs/
|   |-- index.html
|   `-- figures/
|       |-- cohort_outcome_mix.svg
|       |-- formation_curve.svg
|       |-- funding_ladder.svg
|       |-- market_exit_fingerprint.svg
|       `-- status_composition.svg
|-- investments_VC.csv
|-- index.qmd
|-- data.qmd
|-- results.qmd
|-- conclusion.qmd
|-- requirements.txt
`-- README.md
```

## Reproduce The Report

Create a Python environment and install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the static report and figures:

```bash
python scripts/build_publication_report.py
```

The generated report is written to `docs/index.html`, which is served by GitHub Pages.

## Data

The analysis uses a Crunchbase-style startup investment dataset from Kaggle / public redistributed Crunchbase data. It includes company status, market, geography, funding rounds, total disclosed funding, and founding year.

Important limitations:

- The dataset is a static snapshot, not a live Crunchbase feed.
- Startup outcomes are censored because many firms are still operating.
- Funding is aggregated at the company level, not modeled as a time series.
- Missingness and self-reporting bias are substantial.
- The report is descriptive and does not make causal claims.

These are descriptive observations from the available dataset, not causal claims.

## Portfolio Positioning

This project is best presented as an EDA and visualization case study:

- analytical framing
- data quality discussion
- custom visual storytelling
- careful interpretation under bias
- reproducible static reporting

It is intentionally lighter than the ML research repositories, but now reads as a polished data analysis artifact rather than a basic classroom visualization project.
