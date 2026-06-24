# Rtivity-Python

![tests](https://github.com/ChiaLungChuang/Rtivity-Python/actions/workflows/tests.yml/badge.svg)

A Python rewrite of [Rtivity](https://github.com/Rilva/Rtivity) (Silva et al., 2022), a tool for analyzing *Drosophila* activity, sleep, circadian rhythms, and survival from Drosophila Activity Monitor (DAM) data. Built with [Shiny for Python](https://shiny.posit.co/py/), pandas, scipy, and lifelines.

This is not a line-for-line port. It is a reimplementation of the same scientific tool with modern Python tooling, modular architecture, and 229 unit tests covering mathematical invariants of every analysis function.

## Scope

This rewrite covers the four core analysis modules of the original Rtivity and extends them with additional statistical methods. Some visualization features from the R version are not yet implemented — see "What's not here yet" below.

### Implemented

**Activity analysis** (`modules/analysis/activity.py`)
- Activity by phase, day, day×phase, and custom time bins
- Bout detection with duration, count, and intensity metrics
- Summary statistics per group with SEM

**Sleep analysis** (`modules/analysis/sleep.py`)
- Sleep annotation (5 min inactivity rule, *Drosophila* convention)
- Total sleep time (TST), wake after sleep onset (WASO), sleep latency
- Sleep bout count and duration, stratified by phase, day, or day×phase
- Sleep ratio by phase, day, and day×phase

**Rhythm analysis** (`modules/analysis/rhythm.py`)
- Chi-squared periodogram (Sokolove-Bushell)
- Lomb-Scargle periodogram (for unevenly-sampled data)
- Cosinor regression with amplitude, acrophase, and R²
- Non-parametric circadian indices: IS (interdaily stability), IV (intradaily variability), RA (relative amplitude)
- Dominant period detection per animal

**Survival analysis** (`modules/analysis/survival.py`)
- Death detection from prolonged inactivity (configurable threshold)
- Kaplan-Meier curves with confidence intervals (via lifelines)
- Pairwise and multi-group log-rank tests
- Median survival per condition with confidence intervals

### What's added vs the R version

- **WASO** (wake after sleep onset) — standard sleep metric, not in original
- **Lomb-Scargle periodogram** — handles unevenly-sampled or gappy data
- **Cosinor regression** — amplitude, acrophase, R² fitting
- **IS / IV / RA** — non-parametric circadian indices used in human actigraphy
- **Kaplan-Meier with CI + log-rank tests + median survival** — proper biostatistics workflow via lifelines
- **229 unit tests** with property-based assertions (e.g., survival functions are non-increasing, p-values are in [0,1], cosinor amplitudes are non-negative)
- **Continuous integration** — tests run on every push across Python 3.11, 3.12, and 3.13

### What's not here yet

The following features exist in the [R version](https://github.com/ChiaLungChuang/Rtivity-DAM-sleep-analysis) but are not yet in this Python rewrite:

- **Actogram visualizations** — per-channel activity heatmaps across multiple days
- **Chronograms** — multi-day rhythm overlay plots
- **DFA (Detrended Fluctuation Analysis)** — included in the R version via `nonlinearTseries`
- **Bin converter utility** — small standalone tool
- **Data integrity / missing-value imputation tab** — the R version's first tab

If you need these, use the R version. Visualization features will be added in v1.0.

## Installation

Requires Python 3.11 or later.

    git clone https://github.com/ChiaLungChuang/Rtivity-Python.git
    cd Rtivity-Python
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Running

### Shiny app

    shiny run app.py --reload

Then open http://127.0.0.1:8000 in a browser.

### Tests

    pytest tests/ -v

All 229 tests should pass.

## Architecture

    Rtivity-Python/
    ├── app.py                          # Shiny entry point
    ├── requirements.txt
    ├── pyproject.toml
    ├── modules/
    │   ├── data/
    │   │   ├── dam_parser.py           # DAM .txt file parsing (timezone-safe)
    │   │   ├── metadata.py             # Metadata sheet validation
    │   │   └── data_store.py           # Reactive data container
    │   ├── analysis/
    │   │   ├── activity.py             # Activity metrics, bout detection
    │   │   ├── sleep.py                # Sleep scoring, TST, WASO, latency
    │   │   ├── rhythm.py               # Periodograms, cosinor, IS/IV/RA
    │   │   └── survival.py             # Death detection, KM, log-rank
    │   └── ui/
    │       └── plots.py                # Plotly figure generators
    └── tests/                          # 229 unit tests
        ├── test_dam_parser.py
        ├── test_metadata.py
        ├── test_data_store.py
        ├── test_activity.py
        ├── test_sleep.py
        ├── test_rhythm.py
        └── test_survival.py

Analysis functions are pure (DataFrame in → DataFrame out) and have no Shiny dependencies, so they are independently testable and reusable in Jupyter notebooks or other contexts.

## Relationship to other Rtivity versions

| Repo | Language | Status |
|---|---|---|
| [Rilva/Rtivity](https://github.com/Rilva/Rtivity) | R / Shiny | Original by Silva et al. (2022). Unmaintained. |
| [Rtivity-DAM-sleep-analysis](https://github.com/ChiaLungChuang/Rtivity-DAM-sleep-analysis) | R / Shiny | macOS port and R 4.5+ maintenance fork |
| **Rtivity-Python** | Python / Shiny for Python | Reimplementation — this repo |

## Citation

Please cite the original publication:

> Silva, R.F.O., Pinho, B.R., Monteiro, N.M. et al. (2022). Automated analysis of activity, sleep, and rhythmic behaviour in various animal species with the Rtivity software. *Scientific Reports*, 12, 4179. https://doi.org/10.1038/s41598-022-08195-z

If your work specifically uses methods from this rewrite (Lomb-Scargle, cosinor, IS/IV/RA, lifelines-based survival), please additionally cite this repository:

> Chuang, C.-L. (2026). *Rtivity-Python — Drosophila DAM analysis in Python with Shiny* (Version 0.9.0). Demontis Lab, St. Jude Children's Research Hospital. https://github.com/ChiaLungChuang/Rtivity-Python

## License

CC BY 4.0, matching the original Rtivity license.

## Acknowledgments

Original Rtivity authored by Silva et al. (2022). Python reimplementation by [Chia-Lung Chuang](https://github.com/ChiaLungChuang), Department of Developmental Neurobiology, St. Jude Children's Research Hospital (Demontis Lab).