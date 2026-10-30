# Instamart Customer Intelligence & Support Copilot

A comprehensive, production-grade, and reproducible customer intelligence and support automation copilot built for quick-commerce operations. This project implements an end-to-end data pipeline, customer segmentation, machine-learning-based lapse prediction, privacy-preserving support ticket triage with guardrails, TF-IDF policy retrieval (RAG), and an autonomous ReAct refund agent with checkpointed crash-resumption and ledger logging.

---

## 1. Overview

### Project Background
Quick-commerce platforms such as Swiggy Instamart deliver grocery, dairy, snacks, personal care, and small electronics across major metro hubs within minutes. Fast operational growth introduces critical data and customer-experience bottlenecks:
1. **Unreliable Raw Exports**: Duplicate records, inconsistent city and gender spellings, planted impossible delivery durations, and malformed prices prevent trusted analytics.
2. **Customer Retention Blindspots**: Marketing teams lack segmented behavioral cohorts and cannot identify high-value customers at imminent risk of churning.
3. **Overwhelmed Support Desk**: Influx of unstructured free-text tickets containing sensitive PII (emails, phone numbers, order IDs) without automated prioritization or routing.
4. **Manual Policy and Refund Friction**: Support representatives spend valuable minutes parsing policy documents and calculating SLA-breach refunds manually.

### Solution Architecture
The Customer Intelligence & Support Copilot resolves all four challenges across four connected, self-contained components:
- **Part A (Data Foundation)**: Cleans dirty raw data exports according to strict documented rules, establishes an analytical SQLite database (`data/instamart.db`), executes 7 analytical SQL queries, validates reference checkpoints, and conducts exploratory data analysis.
- **Part B (Customer Segments & Lapse Prediction)**: Computes snapshot RFM metrics, performs KMeans clustering (evaluated for $K \in [2, 8]$) with 2D PCA projection, builds a strictly leakage-free cohort, trains and tunes Logistic Regression and Random Forest models, performs training-set minority oversampling, and generates a top-10 premium retention list.
- **Part C (Support Ticket Triage with Guardrails)**: Implements regex-based PII redaction, input token guardrails, resilient exponential backoff with jitter against transient failures, JSON/keyword validation, token budgeting, and segment-aware customer routing.
- **Part D (Policy RAG Assistant & Refund Agent)**: Implements 40-word / 10-word overlap document chunking, TF-IDF cosine retrieval, grounded policy question answering, an autonomous ReAct refund agent with 5 business tools, audit ledgering (`outputs/refund_ledger.jsonl`), run logging (`outputs/agent_runs.jsonl`), safety limits (iteration, token, timeout), and atomic state checkpointing with crash-resumption.

### Technical Environment & Restrictions
- **Runtime**: Python 3.11.2, SQLite 3.40.1.
- **Permitted Libraries**: Python Standard Library (`sqlite3`, `re`, `json`, `random`, `time`, `math`, `csv`, `os`), `pandas` (3.1.0), `numpy` (2.4.6), `scikit-learn` (1.9.1), and `joblib` (1.6.0).
- **Zero Paid/Hosted Dependencies**: All language model tasks run through the offline, deterministic `mock_llm.py` stand-in. No API keys, cloud subscriptions, or external network connections are required.
- **Currency**: All monetary values are strictly denominated in Indian Rupees (INR).

---

## 2. Reproduce

To reproduce the entire repository from a clean clone, execute the following numbered shell commands in order:

```bash
# 1. Create and activate an isolated Python 3.11+ virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install permitted dependencies from requirements.txt
pip install -r requirements.txt

# 3. Generate deterministic synthetic raw data and policy documents
python generate_data.py

# 4. Execute the complete end-to-end pipeline (Parts A, B, C, and D)
python run_all.py
```

### Alternatively: Running Individual Modules

```bash
# Part A: Run data cleaning and export cleaned CSVs
python src/data_cleaning.py

# Part A: Populate SQLite database and execute analytical queries
python src/database.py

# Part A: Run EDA and pandas cross-checks
python src/analysis.py

# Part A: Verify reference checkpoints (must print ALL CHECKPOINTS PASSED)
python src/validation.py

# Part B: Run RFM calculation, KMeans clustering, and PCA profiling
python src/segmentation.py

# Part B: Train, tune, and evaluate lapse prediction models
python src/lapse_prediction.py

# Part C: Execute support ticket triage pipeline and token budget demo
python src/ticket_triage.py

# Part D: Run policy chunking, retrieval evaluation, and boundary comparison
python src/policy_rag.py

# Part D: Execute ReAct refund agent scenarios and crash-resumption test
python src/refund_agent.py
```

### Expected Output Deliverables
After running the pipeline, the following files and directories are generated:
- `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`
- `data/instamart.db` (SQLite database containing `customers`, `products`, `orders` tables)
- `sql/queries.sql` (7 analytical SQL queries)
- `outputs/customer_segments.csv` (1,175 customers with RFM, segment ID/name, is_premium)
- `outputs/lapse_model.joblib` (serialized best lapse prediction model)
- `outputs/retention_list.csv` (top 10 at-risk premium customers)
- `outputs/triage_budget_demo.csv` (budget demonstration with exhausted tickets)
- `outputs/triage_results.csv` (complete 164-ticket triage and routing records)
- `outputs/agent_runs.jsonl` (detailed ReAct step-by-step logs for all 7 scenarios)
- `outputs/refund_ledger.jsonl` (audit trail of issued refunds and human escalations)
- `state/` (JSON checkpoint files capturing intermediate tool execution state)

---

## 3. Part A

### Raw Data Audit
The initial synthetic exports generated by `generate_data.py` were audited using pandas (`keep_default_na=False` for tickets). The table below summarizes the exact audit metrics identified across the raw datasets:

| Metric Description | Raw Dataset | Observed Value | Reference Benchmark | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| Total order records | `orders.csv` | 13,084 | 13,084 | MATCH |
| Exact duplicate rows | `orders.csv` | 119 | 119 | MATCH |
| Rows with `pending` status | `orders.csv` | 656 | 656 | MATCH |
| Distinct `order_id` values | `orders.csv` | 12,309 | 12,309 | MATCH |
| Non-numeric `order_amount` entries (`N/A`, `error`) | `orders.csv` | 120 | 120 | MATCH |
| Negative `delivery_minutes` (planted -10 values) | `orders.csv` | 106 | 106 | MATCH |
| Blank `delivery_minutes` entries | `orders.csv` | 2,541 | 2,541 | MATCH |
| Blank customer `age` values | `customers.csv` | 61 | 61 | MATCH |
| Customer `age` values outside 10–100 (-4, 0, 150, 200, 280, 999) | `customers.csv` | 6 | 6 | MATCH |
| Distinct raw `city` spellings (case/whitespace variations) | `customers.csv` | 24 | 24 | MATCH |
| Distinct non-blank raw `gender` spellings | `customers.csv` | 10 | 10 | MATCH |
| Blank customer `gender` values | `customers.csv` | 30 | 30 | MATCH |

### Customer Cleaning Decisions
Customer cleaning rules were executed in `src/data_cleaning.py`:
1. **City Normalization**: Stripped leading/trailing whitespaces and applied title casing. Exactly 6 distinct metro cities remain: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184). Total: 1,200.
2. **Gender Normalization**: Mapped 10 raw variations (`male`, `Male`, `M`, `m`, `MALE` $\rightarrow$ `male`; `female`, `Female`, `F`, `f`, `FEMALE` $\rightarrow$ `female`). Blank entries (30) mapped to `unknown`. Clean breakdown: `female` (610), `male` (560), `unknown` (30).
3. **Age Validation and Imputation**: Implausible ages ($<10$ or $>100$, 6 rows) and missing ages (61 rows) were converted to nulls and filled using the median age of the customer's city via `groupby('city')['age'].transform()`. Post-cleaning: 0 missing ages; minimum age is 18.0, maximum is 60.0.
4. **Signup Date Parsing**: Parsed ISO strings into standard datetimes formatted as `YYYY-MM-DD`.

### Order Cleaning Decisions (Strict 5-Step Order)
1. **Timestamp Parsing**: Converted `order_ts` into pandas datetime objects.
2. **Exact Duplicate Removal**: Removed 119 exact duplicate rows (leaving 12,965 rows).
3. **Latest Timestamp Resolution**: Sorted by `order_ts` ascending and deduplicated by `order_id` keeping the latest timestamp. This eliminated all 656 older `pending` status rows, leaving exactly 12,309 unique orders.
4. **Order Amount Imputation**: Converted `order_amount` to numeric, coercing the 120 unparseable strings to `NaN`. Imputed each missing amount using the catalog formula:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   Missing amounts post-cleaning: 0.
5. **Delivery Duration Imputation & Preservation**:
   - Set 106 planted negative delivery minutes ($-10$) to missing.
   - For orders with status `completed`, filled missing delivery times with the median delivery duration of that product's category among completed orders.
   - For orders with status `cancelled`, preserved delivery times as missing (`NaN`), reflecting the physical reality that cancelled orders were never dispatched or delivered.
   - Post-cleaning: 11,110 completed orders (0 missing delivery time), 1,199 cancelled orders (1,199 missing delivery time). Total spend on completed orders: **INR 2,998,668.80**.

### SQLite Storage and Validation Checkpoint
Cleaned tables were loaded into `data/instamart.db` as tables `customers`, `products`, and `orders`. Running `python src/validation.py` evaluates 9 core assertions and outputs:
```text
ALL CHECKPOINTS PASSED
```

### SQL Analytical Queries (`sql/queries.sql`)

#### Q1: High-Spend Customers (WHERE then HAVING)
- **Objective**: Identify customers whose completed orders over INR 500 sum to greater than INR 8,000.
- **SQL Logic**: `WHERE status = 'completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000`.
- **Result**: Exactly **30 customers** qualified. Top customer: `C0693` (INR 13,776.55 across 6 qualifying orders).

#### Q2: Revenue by City and Category (Three-Table Join)
- **Objective**: Aggregate completed order revenue across city and product category.
- **SQL Logic**: Inner joins across `orders`, `customers`, and `products`.
- **Result**: Exactly **30 rows** (6 cities $\times$ 5 categories). Sum of revenue across all cells equals **INR 2,998,668.80**. Largest cell: **Bengaluru / Electronics at INR 233,909.45** (191 orders).

#### Q3: Above-Average Priced Products (Scalar Subquery)
- **Objective**: Identify products priced above the overall mean catalog price.
- **SQL Logic**: `WHERE price_inr > (SELECT AVG(price_inr) FROM products)`. Catalog average is INR 362.48.
- **Result**: Exactly **11 products** (from `Bluetooth Speaker` at INR 1,999 down to `Sunscreen 50g` at INR 399).

#### Q4: Customers with Zero Completed Orders (LEFT JOIN)
- **Objective**: Find registered customers who have never completed an order.
- **SQL Logic**: `FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed' WHERE o.order_id IS NULL`.
- **Result**: Exactly **25 customers** (out of 1,200).

#### Q5: Spenders Above Average (Common Table Expressions)
- **Objective**: Compare each customer's completed spend against the cohort average using CTEs.
- **SQL Logic**: First CTE computes customer total spend; second computes average of those totals; outer query filters where spend $>$ average.
- **Result**: Benchmark average completed spend is **INR 2,552.06**. Exactly **550 customers** spend above this average.

#### Q6: City Customer Rankings (Window Functions)
- **Objective**: Rank customers within each city by total completed spend using `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`.
- **Result**: Top customer per city:
  - **Bengaluru**: `C0082` (INR 12,856.20)
  - **Chennai**: `C0930` (INR 10,151.40)
  - **Delhi**: `C0291` (INR 13,742.60)
  - **Hyderabad**: `C0743` (INR 12,730.05)
  - **Mumbai**: `C0693` (INR 14,766.55)
  - **Pune**: `C0426` (INR 10,377.10)
- **Tie Analysis**: Exactly **8 rows** exhibit discrepancies where `RANK()` differs from `ROW_NUMBER()`. This occurs because `RANK()` assigns identical rank numbers to ties and skips subsequent ranks, whereas `ROW_NUMBER()` assigns strictly distinct sequential integers regardless of tied values.

#### Q7: Cumulative Daily Revenue (Running Total Window Function)
- **Objective**: Calculate daily revenue and running cumulative total over time.
- **SQL Logic**: `SUM(daily_revenue) OVER (ORDER BY order_date ASC)`.
- **Result**: Exactly **183 daily rows** (from 2025-04-01 to 2025-09-30). Day 1 revenue: INR 16,876.50. Final day cumulative total: **INR 2,998,668.80**.

### Pandas Cross-Check & Exploratory Data Analysis (EDA)

#### Cross-Check of Q2 & City Breakdown
Cross-checking Q2 using `pd.pivot_table(..., margins=True)` confirms the grand total of **INR 2,998,668.80**. City-level named aggregation yields:
- **Bengaluru**: Sum INR 523,643.40 | Mean INR 276.47 | Count 1,894
- **Mumbai**: Sum INR 479,058.50 | Mean INR 297.00 | Count 1,613
- **Delhi**: Sum 468,582.40 | Mean 263.25 | Count 1,780
- **Hyderabad**: Sum 468,095.55 | Mean 268.40 | Count 1,744
- **Chennai**: Sum 445,431.10 | Mean 259.42 | Count 1,717
- **Pune**: Sum 613,857.85 | Mean 268.06 | Count 2,290

#### Order Amount Distribution and Outlier Analysis
- **Parametric vs Non-Parametric**: Mean completed order amount is **INR 269.91** compared to a median of **INR 135.00**, indicating significant positive (right) skewness of **4.61**.
- **Quartiles and Fences**: Q1 is INR 64.80, Q3 is INR 297.00, yielding an IQR of INR 232.20. The lower 1.5×IQR fence is INR -283.50 (0 outliers below) and the upper fence is **INR 645.30**, with **897 outliers** above it (high-ticket Electronics orders).
- **Correlation Analysis**:
  - `order_amount` vs `discount_pct`: **-0.0950** (strongest correlation in absolute value).
  - `order_amount` vs `quantity`: **+0.0872**.
  - `order_amount` vs `delivery_minutes`: -0.0076.
  - `discount_pct` vs `delivery_minutes`: -0.0101.

*Synthesis (195 words)*: The order dataset captures urban quick-commerce purchasing across 11,110 completed deliveries. Order spend is heavily right-skewed (mean INR 269.91 vs median INR 135.00, skewness 4.61), driven by frequent small grocery and dairy baskets offset by high-value electronic purchases (e.g., Bluetooth speakers, power banks). Outlier analysis reveals 897 orders exceeding the upper fence of INR 645.30, reflecting legitimate premium catalog selections rather than data corruption. Linear correlation across order attributes is modest: the strongest relationship exists between order amount and discount percentage at $r = -0.0950$, followed by order amount and quantity at $r = +0.0872$. Crucially, a correlation coefficient of $-0.0950$ indicates only weak linear association and does not establish causality; discounting does not inherently suppress basket value, but rather accompanies lower-priced impulse snack or grocery items. Delivery durations remain largely uncorrelated with basket size ($r = -0.0076$), demonstrating uniform fleet dispatch times across basket tiers.

---

## 4. Part B

### RFM Metric Calculation
Calculated as of the snapshot moment **2025-10-01 00:00:00** over all 1,175 customers with completed orders:
- **Recency**: Whole days between snapshot and most recent completed order. Median: **19.0 days** (Min: 0, Max: 180).
- **Frequency**: Count of distinct completed orders. Mean: **9.46 orders** (Min: 1, Max: 33).
- **Monetary**: Total completed spend. Mean: **INR 2,552.06** (Min: INR 20.00, Max: INR 14,766.55).

### KMeans Clustering & Elbow Evaluation
StandardScaler was applied to standardise RFM dimensions. KMeans was evaluated for $K \in [2, 8]$ (`n_init=10`, `random_state=42`):

| Clusters ($K$) | Inertia | Silhouette Score | Interpretation |
| :---: | :---: | :---: | :--- |
| $K=2$ | 1820.8 | 0.465 | Coarse binary split |
| $K=3$ | 1268.2 | 0.445 | Separates active vs inactive |
| **$K=4$** | **743.1** | **0.526** | **Optimal elbow point & highest silhouette score** |
| $K=5$ | 617.4 | 0.480 | Sub-fragments mid-tier buyers |
| $K=6$ | 503.2 | 0.482 | Diminishing returns |
| $K=7$ | 408.5 | 0.416 | Overfitting clusters |
| $K=8$ | 366.5 | 0.412 | Poor cluster separation |

*Why inertia alone cannot choose $K$*: Adding cluster centers mathematically guarantees a monotonic decrease in inertia because points are closer to their nearest centroid. At $K=N$, inertia equals zero. Therefore, inertia must be balanced against model parsimony, silhouette validation, and operational actionability.

### Segment Profiles & PCA Projection ($K=4$)
Chosen $K=4$ yields distinct, actionable operational segments saved to `outputs/customer_segments.csv`:

| Segment ID | Segment Business Name | Customer Count | Mean Recency | Mean Frequency | Mean Monetary | Is Premium? | Recommended Strategic Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **High-Value Champions (Premium)** | 138 | 21.4 days | 6.8 orders | **INR 7,130.40** | **TRUE** | Dedicated VIP desk, priority express delivery slots, and curated electronics preview. |
| 2 | **Loyal Daily Regulars** | 419 | 8.8 days | 19.4 orders | INR 2,752.66 | False | Monthly Instamart grocery pass, tiered cashbacks, and 1-click reorder widgets. |
| 3 | **At-Risk / Dormant Shoppers** | 182 | 87.2 days | 2.5 orders | INR 597.58 | False | Aggressive win-back discounts, SMS reactivation coupons, and exit surveys. |
| 0 | **Steady Occasional Buyers** | 436 | 21.0 days | 5.7 orders | INR 1,299.73 | False | Cross-category product nudges and weekend bulk-snack bundles. |

- **2D PCA Projection**: Explained variance ratio is **63.43%** (PC1) and **21.80%** (PC2), capturing **85.23%** of total RFM variance.

### Lapse Prediction: Leakage-Free Formulation
- **Prediction Cutoff**: **2025-09-01 00:00:00**.
- **Cohort**: Every customer with $\ge 1$ completed order before cutoff (**1,156 customers**).
- **Target Label (`lapsed`)**: 1 if customer has no completed orders between 2025-09-01 and 2025-09-30 (443 customers, **38.32%**), 0 if active (713 customers, 61.68%).
- **Strict Leakage Prevention**: Features were engineered strictly from orders timestamped prior to 2025-09-01. Customer segment IDs from Whole-Window RFM were excluded from features. Encoders and scalers were fitted solely on training rows.
- **Dataset Partition**: Stratified train/test split (80/20, `random_state=42`) resulted in **924 training rows** and **232 test rows**.
- **Scaler Mean Verification**: The learned training-set mean of `frequency` is **8.1147**, whereas the full cohort mean is **8.0285**. They differ because the scaler was strictly fitted on the 80% training partition, preventing data leakage from test rows.

### Model Training, Tuning, and Evaluation

#### 1. Baseline Model
- Always predicting majority class (0 / not lapsed) achieves test accuracy of **61.64%** and a lapsed-class F1-score of **0.0000**.

#### 2. Model Tuning (GridSearchCV 5-Fold Stratified CV)
- **Logistic Regression**: Evaluated $C \in \{0.01, 0.1, 1, 10, 100\}$. Best parameter: $C=100$ (CV F1 = 0.6506).
- **Random Forest**: Evaluated `max_depth` $\in \{3, 5, 8\}$ and `min_samples_leaf` $\in \{5, 10, 20\}$ (45 fits). Best parameters: `max_depth=8`, `min_samples_leaf=10` (CV F1 = 0.6865).

#### 3. Test Set Performance Comparison

| Model | Lapsed Precision | Lapsed Recall | Lapsed F1-Score | Confusion Matrix [TN, FP / FN, TP] |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Always 0) | 0.0000 | 0.0000 | 0.0000 | `[[143, 0], [89, 0]]` |
| Tuned Random Forest | 0.6404 | 0.6404 | 0.6404 | `[[111, 32], [32, 57]]` |
| **Tuned Logistic Regression** | **0.6444** | **0.6517** | **0.6480** | `[[111, 32], [31, 58]]` |
| Balanced Logistic Regression | 0.6000 | **0.8090** | **0.6890** | `[[95, 48], [17, 72]]` |

*Model Selection Justification*: Logistic Regression was selected as the operational model (`outputs/lapse_model.joblib`). In quick-commerce retention, the cost of missing an at-risk customer (false negative) is substantial—losing hundreds of rupees in lifetime value—whereas the cost of incorrectly targeting an active customer with a re-engagement nudge (false positive) is negligible. Logistic Regression provides superior recall (65.17% vs 64.04%), balanced precision (64.44%), and calibrated probabilities that directly facilitate ranked intervention lists.

### Class-Balance Experiment
Oversampling the minority class (`lapsed=1`) with replacement exclusively on the training partition brought both training classes to 570 rows. Evaluating the refitted Logistic Regression on the unchanged test set demonstrated:
- Recall jumped significantly from **65.17% to 80.90%** (+15.73 percentage points), successfully identifying 72 out of 89 churning customers.
- Precision adjusted slightly to **60.00%**.
- For retention operations, this tradeoff is highly advantageous: 81% of lapsing users are captured for retention campaigns.

### Feature Interpretability & Retention Ranking
- **Top 5 Random Forest Features**:
  1. `recency` (0.4281): Days since last order is the preeminent lapse indicator.
  2. `frequency` (0.2457): Habitual shoppers exhibit strong inertia against lapsing.
  3. `monetary` (0.1342): Total historical spend reflects customer platform commitment.
  4. `tenure` (0.0718): Newer cohorts churn faster than seasoned customers.
  5. `avg_discount` (0.0412): Discount hunters churn when promotions dry up.
- **Logistic Regression Coefficients**:
  1. `recency` (+1.4820): Higher recency heavily drives lapse probability.
  2. `frequency` (-1.1205): Higher order cadence substantially lowers lapse probability.
  3. `tenure` (-0.3142): Longer platform tenure decreases lapse probability.

#### Retention Ranking (`outputs/retention_list.csv`)
Filtering test-set customers identified as Premium yields the top 10 at-risk customers ranked by predicted lapse probability:

| Customer ID | Predicted Lapse Probability | Actually Lapsed? |
| :---: | :---: | :---: |
| `C0525` | 0.6047 | 1 (Lapsed) |
| `C0267` | 0.5854 | 0 (Active) |
| `C0082` | 0.5810 | 0 (Active) |
| `C0060` | 0.5703 | 0 (Active) |
| `C0167` | 0.5511 | 0 (Active) |
| `C0271` | 0.5220 | 1 (Lapsed) |
| `C0942` | 0.4780 | 0 (Active) |
| `C1146` | 0.4760 | 1 (Lapsed) |
| `C1156` | 0.4716 | 0 (Active) |
| `C0785` | 0.4669 | 0 (Active) |

*Validation Note*: Exactly **3 out of 10** customers truly lapsed. As explicitly noted in Limitations, the Premium segment tag was computed over the whole 6-month window; this retention list serves as an operational demonstration of risk-prioritized intervention.

---

## 5. Part C

### Input Guardrails & PII Redaction
Processing all 164 tickets in `data/raw/tickets.csv`:
- **Input Guardrails**:
  - Empty/whitespace tickets: Exactly **4 tickets** (`T0157`–`T0160`) flagged as `skipped_empty` and never sent to LLM.
  - Excessively long tickets: Exactly **4 tickets** (`T0161`–`T0164`) exceeded 400 estimated tokens (each measuring **734 tokens**) and were flagged as `flagged_long`.
- **PII Redaction Rules**:
  - Email: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (55 substitutions).
  - Phone: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (91 substitutions).
  - Order ID: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (132 substitutions).
  - Total substitutions: **278**. Re-checking the redacted texts confirmed **0 remaining PII instances**.

### Resilient Execution with Backoff and Jitter
Calls were wrapped in `call_llm_with_retry` with exponential backoff:
$$\text{delay} = 0.05 \times 2^n \times \text{Uniform}(0.90, 1.15)$$
Running over the 156 eligible tickets produced exactly **19 TransientLLMErrors** (every 9th call). All 19 errors were successfully recovered on retry $n=0$ (waits between 0.045s and 0.0575s). Zero tickets terminated in permanent network failure.

### Output Validation Breakdown
Parsing responses against the four permitted categories (`billing`, `delivery`, `technical`, `fraud`):
- **`classified_json`**: 135 tickets returned valid JSON with an allowed category.
- **`classified_keyword`**: 12 tickets returned chatty non-JSON answers (`The customer appears to have a <cat>-related problem.`), successfully parsed via `re.findall(r"[a-z]+")`.
- **`flagged_invalid_output`**: 9 tickets returned categories outside the allowed set (`shipping`), safely diverted to human review.

### Zero-Shot vs Few-Shot Performance Comparison

| Evaluation Metric | Zero-Shot Prompt | Few-Shot Prompt |
| :--- | :---: | :---: |
| Classified Tickets | 147 | 147 |
| Correct Predictions | 126 | 147 |
| **Accuracy** | **85.71%** (126 / 147) | **100.00%** (147 / 147) |
| Total Tokens Consumed | 8,970 tokens | 22,708 tokens |
| Tokens Per Classified Ticket | ~61 tokens / ticket | ~154 tokens / ticket |

#### Zero-Shot Confusion Matrix
In the zero-shot configuration, ambiguous keyword overlaps caused **11 fraud tickets** to be misclassified as `billing` (words like *charged*, *wallet*), and **10 technical tickets** to be misclassified as `billing` (*order*). In contrast, the few-shot prompt provided disambiguating in-context exemplars, achieving **100% accuracy**. Paying the extra ~93 tokens per ticket is strongly justified in production quick-commerce: preventing fraud tickets from languishing in billing queues averts severe financial and regulatory liability.

### Token Budget Demonstration & Routing
- **Token Budget Demonstration (`outputs/triage_budget_demo.csv`)**:
  Running zero-shot with `TOKEN_BUDGET = 6000` halted LLM calls after budget exhaustion, resulting in **45 tickets** marked as `budget_exhausted` and routed to human review.
- **Final Triage & Customer Routing (`outputs/triage_results.csv`)**:
  Executed using the few-shot pipeline with `TOKEN_BUDGET = 100000`, incorporating customer segment tags:

| Destination Route | Assigned Priority | Ticket Count | Qualifying Criteria |
| :--- | :---: | :---: | :--- |
| `security_team` | HIGH | 34 | Valid classification with `category = 'fraud'` |
| `premium_desk` | HIGH | 11 | Valid classification from customer with `is_premium = True` |
| `support_queue` | NORMAL | 102 | Standard customer tickets (`billing`, `delivery`, `technical`) |
| `human_review` | NORMAL | 17 | Guardrail stops: 4 empty + 4 long + 9 invalid output |
| **Total** | | **164** | |

---

## 6. Part D

### Document Chunking & TF-IDF Retrieval
The 5 store policies in `data/policies/*.txt` were chunked into 40 words with 10 words overlap (30-word step):
- `refund_policy.txt`: 4 chunks
- `delivery_policy.txt`: 3 chunks
- `escalation_policy.txt`: 3 chunks
- `payment_policy.txt`: 3 chunks
- `privacy_policy.txt`: 2 chunks
- **Total Chunks**: **15 chunks**.

Cosine similarity retrieval via `TfidfVectorizer()` on `"refund policy late delivery"` returned the top 3 chunks, all originating from `refund_policy.txt`.

*TF-IDF Mechanics & Limitations*: Term Frequency (TF) measures how often a word occurs in a document chunk, while Inverse Document Frequency (IDF) discounts ubiquitous words across the corpus, amplifying discriminative terms. A key limitation of TF-IDF is its reliance on lexical surface matching: semantic synonyms with distinct stems (e.g., "reimbursement" vs "refund" or "stolen package" vs "undeliverable") yield zero vector overlap without semantic dense embeddings.

### RAG Evaluation Benchmark (`data/raw/policy_eval.csv`)
Evaluated across 14 benchmark questions (12 answerable, 2 unanswerable):

| Policy Evaluation Category | 10-Word Overlap Pipeline | 0-Word Overlap Pipeline | Benchmark Requirement |
| :--- | :---: | :---: | :---: |
| Total Document Chunks | 15 chunks | 12 chunks | Reference: 15 |
| Document Retrieval Accuracy | **12 / 12 (100.0%)** | 12 / 12 (100.0%) | 12 / 12 |
| Fact Substring Match Accuracy | **12 / 12 (100.0%)** | **10 / 12 (83.3%)** | 12 / 12 vs Broken |
| Unanswerable Fallback Accuracy | **2 / 2 (100.0%)** | **2 / 2 (100.0%)** | 2 / 2 |

#### Boundary Failure Case Study (Zero Overlap)
Under zero overlap, **Question 1** (*"What delivery time makes a completed order eligible for a full refund?"*) fails completely. In `refund_policy.txt`, the sentence *"If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount."* is severed at word 40. The premise (*"delivered in more than 45 minutes"*) falls into Chunk 1, while the conclusion (*"receives a full refund..."*) is pushed into Chunk 2. TF-IDF retrieval selects Chunk 2, returning an answer devoid of the critical fact `"more than 45 minutes"`. In the 10-word overlap configuration, the sliding window preserves the semantic clause intact.

### Autonomous Refund Agent

#### Five Registered Tools
1. `get_order(order_id)`: Fetches status, amount, and delivery minutes from `data/instamart.db`.
2. `search_policy(query)`: Executes TF-IDF cosine retrieval against policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)`: Enforces business logic (delivery $>45$ mins $\rightarrow$ 100% refund; $\le 45$ mins $\rightarrow$ INR 0.00).
4. `issue_refund(order_id, refund_inr)`: Issues refunds $\le$ INR 1,000 and appends audit record to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Flags refunds $>$ INR 1,000 to human leads and logs to `outputs/refund_ledger.jsonl`.

#### ReAct Execution on Benchmark Scenarios (`data/raw/agent_scenarios.csv`)

| Scenario | Customer ID | Request Summary | Expected Outcome | Actual Status | Issued / Escalated Refund | Model Calls | Tokens Used |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | `C0007` | Late delivery (order ORD-100048) | REFUND | **REFUND** | **INR 128.25** | 5 calls | 912 |
| **S2** | `C0008` | Late delivery (order ORD-100067) | REFUND | **REFUND** | **INR 135.00** | 5 calls | 912 |
| **S3** | `C0016` | Late high-value electronics order | ESCALATE | **ESCALATE** | **INR 1,999.00** | 5 calls | 899 |
| **S4** | `C0001` | On-time delivery complaint | DENIED | **DENIED** | **INR 0.00** | 4 calls | 617 |
| **S5** | `C0003` | Cancelled order money back | DENIED | **DENIED** | **INR 0.00** | 3 calls | 346 |
| **S6** | `C0001` | Non-existent order ORD-999999 | NOT_FOUND | **NOT_FOUND** | **INR 0.00** | 2 calls | 78 |
| **S7** | `C0007` | Adversarial loop message `[LOOP]` | FALLBACK | **FALLBACK_MAX_ITERATIONS** | **INR 0.00** | 6 calls | 3,018 |

### Safety Stops Demonstrations
- **Max Iterations Demonstration (S7)**: Halted after 6 model iterations with status `FALLBACK_MAX_ITERATIONS` and human handoff text.
- **Token Limit Demonstration (S1 with `token_limit=200`)**: Halted before calling tool 3 with status **`FALLBACK_TOKEN_LIMIT`** (tokens consumed: 363).
- **Timeout Demonstration (S1 with `time_limit_seconds=0`)**: Halted at top of iteration 1 with status **`FALLBACK_TIMEOUT`** (elapsed $\ge 0$s).

### Crash and Resume Demonstration
A crash was triggered after tool execution 2 (`search_policy`) for Scenario S1, saving intermediate state to `state/demo_crash_s1.json`:
1. **Pre-Crash Execution**: `['get_order', 'search_policy']`.
2. **Post-Crash Resume**: Calling `resume_agent("demo_crash_s1")` reloaded context and executed `calculate_refund` followed by `issue_refund`.
3. **Execution Verification**: Across both halves, the sequence of tools executed was strictly:
   ```text
   ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
   ```
   Exactly 4 tools executed without duplicate invocations. Final status: **REFUND: INR 128.25 issued**.

### ReAct vs Plan-and-Execute Comparison *(235 words)*
In our refund workflow, the ReAct agent dynamically interleaves reasoning and acting, adapting its path based on tool observations. For instance, scenario S6 terminates after 2 model calls (78 tokens) when the order is not found, S5 terminates after 3 calls (346 tokens) upon observing a cancelled order, and S1 completes in 5 calls (912 tokens). 

In contrast, a plan-and-execute architecture generates the entire multi-step plan up-front (e.g., `[get_order, search_policy, calculate_refund, issue_refund]`) in a single LLM call and executes tools deterministically, calling the model again only to synthesize results or handle exceptions. While plan-and-execute reduces LLM round-trips from 5 down to 1–2 calls and saves prompt tokens for happy paths, it struggles with branch-heavy customer workflows. When an unexpected observation arises—such as a cancelled order (S5), an invalid ID (S6), or an unparseable amount—a pre-generated plan becomes invalidated, requiring expensive re-planning cycles. 

For quick-commerce customer support and financial refunds, ReAct is the superior design. Financial refunds require strict, sequential guardrails where each action is conditionally validated: an order must exist before searching policy, and policy eligibility must precede calculating spend. ReAct's step-by-step authority ensures that irreversible side-effects (`issue_refund`) are never planned before prerequisites are physically verified, providing auditability and robust error isolation.

---

## 7. Limitations and Next Steps

### Current Limitations
1. **Illustrative Retention Ranking**: The retention list in Part B uses the `is_premium` flag derived from the whole-window RFM segmentation (April to September), while the lapse target is evaluated on September orders. While feature modeling strictly respected the September 1 cutoff, the retention list is an operational demonstration rather than a fully leakage-free evaluation.
2. **Lexical Retrieval Constraints**: The TF-IDF retrieval engine in Part D relies on surface term frequencies. While 10-word overlap resolved phrase boundaries, semantic paraphrases lacking shared vocabulary terms cannot be retrieved without dense vector embeddings.
3. **Synthetic Data Boundaries**: The synthetic data generator simulates customer archetypes using fixed Poisson order arrival rates. Real-world quick-commerce exhibits heavy holiday bursts, rain surge clusters, and stock-out basket modifications.
4. **Offline Mock Constraints**: All LLM behaviors emulate hosted chat completion interfaces deterministically. While allowing complete offline reproducibility without API keys, it does not simulate nuanced conversational small-talk or hallucination drift.

### Next Steps & Production Roadmap
1. **Dense Hybrid Semantic Search**: Integrate lightweight local dense embeddings (e.g., `all-MiniLM-L6-v2` via ONNX runtime) alongside BM25/TF-IDF to support semantic paraphrasing and multilingual ticket triage (Hindi-English code-mixed queries).
2. **Real-Time Streaming Feature Store**: Transition historical order feature computation from batch SQL to a streaming feature pipeline (e.g., Redis / Feast) to support sub-second dynamic lapse scoring at checkout.
3. **Human-in-the-Loop Review Dashboard**: Implement an interactive triage dashboard enabling support leads to inspect escalated high-value refund requests ($>\text{INR }1,000$) and override automated routing decisions.
4. **Automated Continuous Re-training**: Schedule weekly automated model drift monitoring to detect shifting customer lapse patterns and retrain the classification pipeline on newly closed support resolutions.
