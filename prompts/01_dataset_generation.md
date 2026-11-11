# AI Disclosure 01: Dataset Generation

| Field | Value |
|---|---|
| Date generated | 2026-11-11 |
| Member | Member A (Team Leader & Data Engineer) |
| GenAI tool | Claude (Anthropic) |
| Output files | `src/generate_data.py`, `data/raw/raw_transactions.csv`, `data/raw/typology_key.csv`, `data/WIE3007_AML_Dataset.xlsx`, `docs/WIE3007_AML_Dataset_DataCard.pdf` |
| Random seed | 42 (`np.random.default_rng(42)`) |

## What GenAI was used for

The synthetic AML dataset was generated with help from Claude (GenAI). Claude helped write the generator script `src/generate_data.py` and the data card.

## How the data was produced

1. **Numeric patterns simulated in Python.** Accounts, amounts, timestamps, channels and behavioural features (`txn_count_24h`, `unique_recipients_7d`, `avg_amount_30d`, `account_age_days`) are simulated with NumPy and pandas.
2. **Five planted laundering typologies** make up the suspicious class (`is_suspicious = 1`):
   - **Structuring**: repeated cash deposits just under the RM25,000 reporting threshold within a few days.
   - **Pass-through**: a large inbound amount moved out within hours in several smaller transfers or wires.
   - **Mule**: new low-income accounts receiving several small transfers, then cashing out almost the full total.
   - **High-risk wire**: large outbound wires to high- or medium-risk jurisdictions.
   - **Income mismatch**: transactions of RM15,000 to RM120,000 on accounts with low declared income.
3. **Legitimate look-alikes (hard negatives, label 0)** are added so the task is not trivially easy: cash-heavy businesses banking near-threshold takings, lump sums (property sale, EPF withdrawal) moved out quickly, new online sellers with many small inbound payments, and genuine trade wires to higher-risk countries.
4. **Text pools written by GenAI.** The `transaction_memo` and `kyc_note` template pools (English and Malay) were written by Claude, then sampled in Python with deliberate noise:
   - about 30% of suspicious transactions get ordinary-looking memos;
   - about 8% of normal transactions get vague memos;
   - about 40% of suspicious accounts get a benign KYC note, and about 6% of normal accounts a concerning one.
5. **Data-quality issues for EDA**: some accounts have missing `declared_monthly_income` (12 accounts) or `occupation` (8 accounts).

## Notes

- `typology_key.csv` records which typology produced each row. It is for analysis only and must never be used as a model feature.
- No API keys or personal data were used; all data is synthetic.
