# WIE3007 AML Project: Suspicious Transaction Detection

## Objective

This project detects suspicious bank transactions for Anti-Money Laundering (AML) monitoring, framed as a binary classification problem. We simulate a transaction dataset with planted laundering patterns, use an LLM to extract structured features from the free-text transaction memos and KYC notes, and then train and compare five machine learning models (Decision Tree, Random Forest, XGBoost, Logistic Regression and MLP) on the combined tabular and LLM-extracted features.

## Team

| Member | Role | Model |
|---|---|---|
| Member A | Team Leader & Data Engineer | Decision Tree (baseline) |
| Member B | LLM Feature Engineer | Random Forest |
| Member C | EDA & Evaluation Lead | XGBoost |
| Member D | Interpretation & Report Lead | Logistic Regression + MLP |

## Dataset

A synthetic dataset of **5,000 transactions** across **812 accounts** (January to June 2026), about **4.1% suspicious**, generated with seed 42.

| File | Contents |
|---|---|
| `data/raw/raw_transactions.csv` | The modelling dataset (18 columns, target `is_suspicious`) |
| `data/raw/typology_key.csv` | Which laundering typology or look-alike produced each row |
| `data/WIE3007_AML_Dataset.xlsx` | Excel copy of the dataset for browsing |
| `src/generate_data.py` | Generator script; run `python src/generate_data.py` from the repo root to rebuild `data/raw/` |

Column definitions, planted patterns and noise levels are in the [data card](docs/WIE3007_AML_Dataset_DataCard.pdf).

> **Warning:** `typology_key.csv` is the answer key. Never use it, or the `typology` column, as a model feature. Use it only for analysis after modelling, such as checking which typologies a model misses.

## Folder structure

```
.
├── data/
│   ├── raw/            # Simulated dataset (CSV, committed to git)
│   └── processed/      # Cleaned data and LLM-enriched features
├── notebooks/
│   └── WIE3007_AML_Project.ipynb   # Main project notebook
├── src/
│   ├── generate_data.py    # Dataset simulation
│   ├── llm_features.py     # LLM feature extraction from memos and KYC notes
│   ├── preprocess.py       # Encoding, scaling, train/test split
│   └── evaluate.py         # Metrics and comparison plots
├── prompts/            # Log of every GenAI/LLM prompt used (AI disclosure)
├── results/            # Metric tables and figures
├── report/             # Final report files
├── models/             # Saved models (not committed)
├── data_schema.md      # Dataset columns, planted patterns and project rules
├── CONTRIBUTING.md     # Team git workflow
├── requirements.txt
└── .env.example        # Template for API keys
```

## Setup

1. Clone the repository and move into it.

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   ```bash
   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Copy `.env.example` to `.env` and add your API key. Never commit `.env`.

   ```bash
   cp .env.example .env
   ```

## Running the notebook

With the virtual environment active, start Jupyter from the repository root:

```bash
jupyter notebook notebooks/WIE3007_AML_Project.ipynb
```

Then choose **Kernel > Restart & Run All**. The notebook should run from top to bottom without errors, in section order: dataset simulation, LLM feature extraction, EDA, preprocessing, models, evaluation, interpretation.

## Further reading

- [data_schema.md](data_schema.md): the dataset definition and modelling rules
- [CONTRIBUTING.md](CONTRIBUTING.md): branching, pull requests and commit rules
