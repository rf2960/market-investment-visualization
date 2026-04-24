# Startup Investment Patterns and Outcomes

A Quarto-based exploratory data analysis project that studies how startup outcomes relate to founding cohorts, market segments, and funding intensity using Crunchbase venture data.

## Project Summary

This project analyzes a large startup investment dataset to understand broad patterns in company status, funding, and market composition. The focus is descriptive rather than predictive: the goal is to communicate interpretable trends through clear visualizations and concise narrative analysis.

## Motivation

Startup ecosystems are noisy, uneven, and heavily shaped by time and capital. This project was built to answer simple but meaningful questions:

- Which markets dominate the dataset?
- How do startup outcomes vary across founding cohorts?
- How does funding intensity relate to operating, acquired, or closed status?
- What broad structural patterns appear in venture-backed company trajectories?

## Live Report

Published Quarto site:

- [https://rf2960.github.io/Investments/](https://rf2960.github.io/Investments/)

## Key Features

- End-to-end exploratory analysis in Quarto
- Descriptive visualizations for a non-technical audience
- Clear narrative interpretation alongside plots
- Reproducible report structure with source `.qmd` files and rendered `docs/` output
- Focus on startup outcomes, funding, and market-level heterogeneity

## Tech Stack

- R
- Quarto
- tidyverse
- ggplot2
- janitor
- lubridate
- naniar
- GGally
- ggalluvial

## Repository Structure

```text
.
|-- _quarto.yml
|-- index.qmd
|-- data.qmd
|-- results.qmd
|-- conclusion.qmd
|-- investments_VC.csv
|-- docs/
|   |-- index.html
|   |-- data.html
|   |-- results.html
|   `-- conclusion.html
`-- quarto-edav-template.Rproj
```

## Analysis Workflow

1. Load and clean the Crunchbase startup dataset.
2. Examine missingness and variable quality.
3. Explore startup formation trends over time.
4. Compare market sizes and outcome distributions.
5. Study how funding intensity and cohort timing relate to operating, acquisition, and closure outcomes.
6. Summarize the main descriptive findings and limitations.

## Setup

Prerequisites:

- R
- Quarto
- The R packages used in the `.qmd` files

## How To Run

Render the full report:

```bash
quarto render
```

If you prefer to work from RStudio, open `quarto-edav-template.Rproj` and render the project from there.

## Example Usage

This repository is useful as:

- a course project in EDA and visualization
- a portfolio example of narrative data analysis
- a compact example of Quarto-based reporting

## Results / What This Project Shows

The report highlights several broad patterns:

- most startups in the sample remain operating
- software and biotech dominate the dataset by company count
- newer startup cohorts are more likely to still be operating, partly due to censoring and survivorship effects
- funding depth and market context appear closely tied to startup outcomes

These are descriptive observations from the available dataset, not causal claims.
