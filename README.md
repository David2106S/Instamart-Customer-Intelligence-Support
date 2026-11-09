# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four connected parts of the Instamart Customer Intelligence & Support Copilot capstone: data foundation and SQL analytics, customer segmentation and lapse prediction, guarded support-ticket triage, and a policy RAG assistant with a simulated refund agent. All data are synthetic and fictional. Every model call uses the supplied offline `mock_llm.py`; there are no network calls, API keys, paid services or real refund transactions.

## Reproduce

Run these commands from the repository root using Python 3.11 or newer:

1. `python -m venv .venv`
2. Windows: `.venv\\Scripts\\activate` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

`generate_data.py` and `mock_llm.py` are the starter scripts copied from the supplied assignment brief; do not modify them. The generator should print `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `run_all.py` executes Parts A–D in dependency order. The pipeline writes its cleaned data, SQLite database, SQL outputs, model, CSV outputs, JSON reports, agent logs and checkpoints under `data/`, `outputs/` and `state/`.

## Part A

### Audit and cleaning

The raw data audit reports 1,200 customers, 40 products and 13,084 raw order rows. The raw orders contain 119 exact duplicate rows, 656 pending rows, 12,309 distinct order IDs, 120 unparseable amounts, 106 negative delivery durations and 2,541 blank delivery durations. The customer export contains 61 blank ages, 6 impossible ages, 24 city spellings, 10 nonblank gender spellings and 30 blank genders. Per-column missing counts and percentages are written to `outputs/part_a_audit.json`.

Cleaning rules: city values are stripped and title-cased; gender values are mapped to `male`, `female` or `unknown`; ages outside 10–100 are set missing and missing ages are filled with the city median; signup and order timestamps are parsed; exact duplicate order rows are removed; duplicate order IDs retain the latest timestamp; invalid amounts are recalculated from product price × quantity × (1 − discount/100); negative delivery durations become missing; missing completed-order delivery times are filled with the completed-order median for that product category; cancelled delivery times remain missing.

The clean order table has 12,309 rows: 11,110 completed and 1,199 cancelled, with no pending rows. Completed-order revenue is INR 2,998,668.80. Completed orders have no missing delivery durations; cancelled orders retain 1,199 missing delivery durations. The Part A validation checks these reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas findings

Seven runnable queries are saved in `sql/queries.sql`; the corresponding result exports are `outputs/part_a_q1.csv` through `part_a_q7.csv`. The result row counts are Q1 30 customers, Q2 30 city/category rows, Q3 11 products, Q4 25 customers without completed orders, Q5 550 above-average spenders, Q6 1,175 customers with completed spend, and Q7 183 daily revenue rows. Q7's final cumulative revenue is INR 2,998,668.80. Q2's largest city/category revenue cell is Bengaluru / Electronics at INR 233,909.45.

The pandas pivot-table cross-check is in `outputs/part_a_city_category_pivot.csv`; city aggregates are in `outputs/part_a_city_stats.csv`. Bengaluru completed-order revenue is INR 523,643.40 (1,894 orders; mean INR 276.47) and Mumbai revenue is INR 479,058.50 (1,613 orders; mean INR 297.00). Completed order amounts have mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR upper fence is INR 645.30; there are 897 upper outliers and no lower outliers. The strongest absolute correlation is order amount with discount percentage (about −0.095); this is a very weak association and does not establish causation. Full statistics and the correlation matrix are in `outputs/part_a_eda.json`.

## Part B

### RFM segmentation

RFM uses completed orders and the 2025-10-01 snapshot. It contains 1,175 customers; recency median is 19 days (range 0–180), mean frequency is 9.46 (maximum 33), and mean monetary value is INR 2,552.06 (maximum INR 14,766.55). The K-means scores for K=2–8 are in `outputs/kmeans_scores.csv`. K=4 was selected: it has the best silhouette score among the candidates (about 0.526) and yields interpretable customer groups. Inertia always falls as K increases because adding clusters cannot increase within-cluster squared distance; that does not alone select K because too many clusters can over-segment customers and reduce business usefulness.

The K=4 segment profiles are approximately: Premium 138 customers (mean monetary INR 7,129.70); Frequent Loyalists 419 (mean frequency 19.40, mean recency 8.77 days); Occasional Shoppers 436; and Dormant Customers 182 (mean recency about 109 days). Recommended actions: protect Premium customers with proactive service and tailored loyalty benefits; reward Frequent Loyalists with repeat-purchase bundles; use relevant reminders and category offers for Occasional Shoppers; and test win-back campaigns for Dormant Customers. Cluster IDs are arbitrary labels, not an ordinal ranking. PCA explains about 63.43% and 21.80% of variance in its first two components (85.23% combined). Segment assignments are in `outputs/customer_segments.csv`.

### Lapse prediction

The pre-cutoff cohort contains 1,156 customers, including 443 lapsed (38.32%); the stratified split is 924 training and 232 test rows. The always-not-lapsed baseline has 61.6% accuracy and lapsed-class F1 of 0. The tuned logistic regression test F1 is about 0.648 (precision 0.644, recall 0.652); the tuned random forest test F1 is about 0.640 (precision and recall about 0.640). The logistic regression is selected based on its slightly higher test F1. Grid-search settings and confusion matrices are in `outputs/part_b_report.json`.

The training-only oversampled logistic regression raises recall to about 0.854, with precision about 0.589 and F1 about 0.697. The model and training pipeline are saved in `outputs/lapse_model.joblib`; the script reloads the model and asserts identical test predictions. `outputs/retention_list.csv` contains up to 10 Premium test customers ranked by predicted lapse probability. No order on or after 2025-09-01 is used in lapse features; Part B does not use segment labels as model features. The Premium retention list is illustrative and not a leakage-free evaluation because Premium is a post-hoc label derived from the full order window, including September.

## Part C

The pipeline screens four empty tickets and four long tickets, leaving 156 eligible tickets. It redacts 55 email addresses, 91 phone numbers and 132 order IDs before prompts. Transient errors are retried with exponential backoff and jitter; the reference run recovers all 19 simulated transient errors. The zero-shot run classifies 147 tickets at 85.7% accuracy (126/147); the few-shot run classifies 147 at 100% accuracy. Few-shot uses more tokens (12,364 versus 8,464 in this run), but its accuracy gain is meaningful for sensitive fraud/technical routing. The budget-constrained zero-shot demonstration marks 63 tickets `budget_exhausted` at the 5,000-token budget. Final few-shot routing is saved in `outputs/triage_results.csv` (164 rows); budget output is `outputs/triage_budget_demo.csv`; detailed counts and confusion results are in `outputs/part_c_report.json`.

Routing rules send skipped, long, invalid or budget-exhausted tickets to human review; fraud to the security team at HIGH priority; Premium customers to the premium desk at HIGH priority; and other safely classified tickets to the support queue at NORMAL priority.

## Part D

### Policy retrieval and grounded answers

The five policy files produce 15 chunks with 40 words per chunk and 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `PolicyRAG` fits `TfidfVectorizer` on the chunks and ranks them with cosine similarity. Term frequency (TF) measures how often a term appears in a chunk; inverse document frequency (IDF) gives more weight to terms that are rarer across the chunk collection. A limitation is that TF-IDF relies on word overlap and may miss two semantically equivalent questions phrased with different words.

With overlap 10, all 12 answerable evaluation questions retrieve the expected document and contain the expected fact; both unsupported questions receive the required fallback. With zero overlap, only 10 of 12 answerable answers contain the expected fact. For example, “What delivery time makes a completed order eligible for a full refund?” fails because the qualifying delivery-time rule is split across chunk boundaries. See `outputs/policy_eval_results.json`.

### Refund agent and safety

Five tools are registered in `TOOLS_SPEC` and `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. The seven scenarios produce S1 REFUND INR 128.25 (5 model calls), S2 REFUND INR 135.00 (5), S3 ESCALATE INR 1,999.00 (5), S4 DENIED (4), S5 DENIED (3), S6 NOT_FOUND (2), and S7 FALLBACK_MAX_ITERATIONS (6). Per-run results and tool observations are in `outputs/agent_runs.jsonl`; simulated refund/escalation actions are in `outputs/refund_ledger.jsonl`.

The S1 token-limit demonstration returns `FALLBACK_TOKEN_LIMIT` at a 200-token limit; the zero-second demonstration returns `FALLBACK_TIMEOUT`. The crash/resume test checkpoints after the second tool observation and resumes without repeating completed tools. The tool sequence across the full S1 demonstration is `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, ending with REFUND INR 128.25. Checkpoint JSON files are stored in `state/`.

ReAct uses a model decision after each observation; the reference runs take 5 calls for S1, 3 for S5 and 2 for S6. A plan-and-execute design could request the whole sequence in one call and use fewer tokens/calls on the happy path, but it is less adaptive when an observation is unexpected (for example, an order is cancelled or not found). ReAct is preferable for this refund workflow because it verifies each tool result before taking the next action, especially before issuing or escalating a refund.

## Limitations and Next Steps

TF-IDF depends on lexical overlap and may miss semantically equivalent wording. The deterministic offline mock LLM is not representative of production model quality. Synthetic data and policy files do not represent real customer records or official company policy. Refund ledger entries are simulated and do not move money. The Premium retention list is illustrative because the Premium label is a post-hoc full-window label; the model features themselves exclude all orders from 2025-09-01 onward. Before production use, add human review, monitoring, access controls, secure PII handling, model drift checks, idempotent transaction controls and policy/legal approval.
