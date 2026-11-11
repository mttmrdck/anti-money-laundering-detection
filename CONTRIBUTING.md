# Contributing

These rules apply to all four members. They keep `main` working and make each person's contribution visible in the git history.

## 1. Never push directly to `main`

Each member works on their own branch and merges through a pull request. Name branches after the piece of work:

| Branch | Work |
|---|---|
| `feature/data-simulation` | Dataset generator |
| `feature/llm-features` | LLM feature extraction |
| `feature/eda` | Exploratory data analysis |
| `feature/decision-tree` | Decision Tree baseline |
| `feature/random-forest` | Random Forest |
| `feature/xgboost` | XGBoost |
| `feature/logreg-mlp` | Logistic Regression and MLP |
| `feature/evaluation` | Metrics and model comparison |
| `feature/interpretation` | SHAP and LLM insights |

Create other `feature/...` branches as needed.

```bash
git checkout main
git pull
git checkout -b feature/xgboost
```

## 2. Pull `main` before starting work each week

```bash
git checkout main
git pull
git checkout feature/your-branch
git merge main
```

This keeps your branch close to everyone else's work and makes notebook merge conflicts smaller.

## 3. Open a pull request and get one review

- Push your branch and open a pull request into `main`. Fill in the template.
- At least one other member must review it and leave a real comment before it is merged: a question, a suggestion, or a note on what they checked. "LGTM" alone does not count.
- The author merges after approval, then deletes the branch.

## 4. Write imperative, specific commit messages

Say what the commit does, starting with a verb in the imperative.

| Good | Bad |
|---|---|
| `Add transaction generator with structuring pattern` | `update` |
| `Fix label leakage in GroupShuffleSplit by account_id` | `fix bug` |
| `Plot PR curves for all five models` | `changes` |

Keep commits small: one logical change each.

## 5. Minimum commits

Each member needs at least 6 commits between Weeks 5 and 13. Spread them out across the weeks; do not save them all for the end.

## 6. Log every GenAI/LLM prompt

Every prompt used with a GenAI tool or LLM must be saved in `prompts/` for the AI disclosure. This covers both:

- prompts that are part of the pipeline (for example the feature-extraction prompt), and
- prompts used for help with code, analysis or report writing.

For each prompt, record the date, your name, the tool or model, the prompt text, and what the output was used for.

## 7. Before you open a pull request

- The notebook runs from top to bottom without errors (**Kernel > Restart & Run All**).
- No API keys or `.env` file are committed.
- Any new GenAI prompts are logged in `prompts/`.
