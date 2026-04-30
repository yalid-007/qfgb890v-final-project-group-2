# QFGB890V Final Project — Group 2

**Project 2: Quote Imbalance as a Trading Signal**

Spring 2026, Gabelli School of Business, Fordham University.

## Team
- Atharva Bokil
- Mackenzie Qu
- Mike Li
- Yalid Rahman

## Setup

1. Clone this repo:

git clone git@github.com:yalid-007/qfgb890v-final-project-group-2.git
cd qfgb890v-final-project-group-2

2. Create the conda environment:

conda env create -f environment.yml
conda activate qfgb-final

3. Download the data zip `nbbosz_20250401_20250404.zip` from Blackboard
   ("Extra Materials/data") and extract the parquet files into the `data/` folder.

4. Launch Jupyter:

jupyter notebook code/Group2-FinalProject.ipynb

## Repository Structure

- `code/` — Python modules and the main analysis notebook
- `data/` — local data files (gitignored)
- `tests/` — unit tests
- `figures/` — generated plots for the technical document

## Workflow

- Work on feature branches: `feature/<description>`
- Open a PR into `main` and request a review before merging
- One GitHub issue per task; reference issue numbers in PRs


