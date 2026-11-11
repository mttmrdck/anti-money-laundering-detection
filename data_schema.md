# Data Schema

This document defines the dataset to be simulated for the project. The generator (`src/generate_data.py`) must follow it, and any change to it goes through a pull request.

## Overview

- One row = one transaction.
- About 5,000 transactions across about 800 accounts.
- Target: `is_suspicious` (0/1), about 4% positive (roughly 200 rows).
- Output file: `data/raw/transactions.csv`.

## Columns

| Column | Data type | Description | Expected range / categories |
|---|---|---|---|
| `transaction_id` | string | Unique transaction identifier | `TXN000001` to `TXN005000`, unique |
| `account_id` | string | Account that made the transaction; the grouping key for the train/test split | `ACC0001` to `ACC0800`, about 800 unique |
| `timestamp` | datetime | Date and time of the transaction | `YYYY-MM-DD HH:MM:SS`, within one 6-month window |
| `amount_myr` | float | Transaction amount in Malaysian Ringgit | 1.00 to 500,000.00, right-skewed (most under 5,000) |
| `txn_type` | categorical | Type of transaction | `transfer`, `cash_deposit`, `cash_withdrawal`, `wire`, `bill_payment`, `card_payment` |
| `channel` | categorical | Channel used | `online_banking`, `mobile_app`, `atm`, `branch` |
| `is_cross_border` | int (binary) | Whether the destination is outside Malaysia | 0 = domestic, 1 = cross-border |
| `dest_country_risk` | categorical | AML risk tier of the destination country | `low`, `medium`, `high`; always `low` for domestic transactions |
| `account_age_days` | int | Age of the account at the time of the transaction | 1 to 7,300 (up to about 20 years) |
| `txn_count_24h` | int | Number of transactions by this account in the previous 24 hours, including this one | 1 to 50; typically 1 to 5 |
| `unique_recipients_7d` | int | Distinct recipients this account sent money to in the previous 7 days | 0 to 40; typically 0 to 5 |
| `avg_amount_30d` | float | Account's average transaction amount over the previous 30 days (MYR) | 1.00 to 100,000.00 |
| `occupation` | categorical | Declared occupation of the account holder | `student`, `teacher`, `engineer`, `civil_servant`, `retail_worker`, `business_owner`, `freelancer`, `doctor`, `retiree`, `unemployed`, `money_changer`, `trader` |
| `declared_monthly_income` | float | Monthly income declared at onboarding (MYR) | 0.00 to 50,000.00 |
| `transaction_memo` | text | Free-text reference entered by the customer | Short text, about 2 to 15 words, e.g. "rent for March" |
| `kyc_note` | text | Bank officer's free-text note from customer due diligence | 1 to 3 sentences on source of funds, behaviour and documents |
| `is_suspicious` | int (binary) | Target label | 0 = normal, 1 = suspicious; about 4% positive |

## Planted laundering patterns

- **Structuring**: several cash deposits just below the RM25,000 reporting threshold (for example RM22,000 to RM24,900) made by the same account within a few days.
- **Rapid pass-through**: a large incoming amount followed within hours by outgoing transfers of nearly the same total, giving a high `txn_count_24h` and an `amount_myr` far above `avg_amount_30d`.
- **Mule accounts**: new accounts (low `account_age_days`) held by low-income occupations that suddenly send to many recipients (high `unique_recipients_7d`).
- **High-risk jurisdiction wires**: `txn_type = wire` with `is_cross_border = 1` and `dest_country_risk = high`, for large amounts and with vague memos.
- **Income mismatch**: `amount_myr` many times larger than `declared_monthly_income`, for example a student or retail worker moving RM80,000.

## LLM-extracted features (added later by Member B)

These columns are produced from the text columns and saved to `data/processed/`.

| Feature | Source | Type | Description | Expected values |
|---|---|---|---|---|
| `memo_category` | `transaction_memo` | categorical | Stated purpose of the transaction | `salary`, `bills`, `family_support`, `business_payment`, `loan`, `investment`, `gift`, `unclear` |
| `memo_vagueness` | `transaction_memo` | ordinal int | How vague or evasive the memo is | 0 = specific, 1 = somewhat vague, 2 = very vague |
| `occupation_risk` | `occupation`, `kyc_note` | ordinal | AML risk associated with the occupation (cash-intensive or hard-to-verify income scores higher) | `low`, `medium`, `high` |
| `kyc_risk_tone` | `kyc_note` | ordinal | Level of concern expressed in the officer's note | `neutral`, `cautious`, `concerning` |
| `customer_segment` | `kyc_note`, `occupation` | categorical | Customer type inferred from the text | `student`, `salaried`, `self_employed`, `business`, `retiree`, `high_net_worth` |

## Rules

1. **Split by account, not by row.** Use `GroupShuffleSplit` with `groups=account_id`, an 80/20 split and `random_state=42`, so that no account appears in both train and test.
2. **The LLM never sees the label.** It receives the text columns only (`transaction_memo`, `kyc_note`, and `occupation` where needed), never `is_suspicious` or anything derived from it.
3. **Label noise in the text.** About 30% of suspicious rows get normal-looking memos, so the text features help but cannot identify the label on their own.
4. **Metrics.** The main metrics are Recall, Precision, F1, PR-AUC and ROC-AUC. Accuracy is not a main metric because only about 4% of rows are positive.
