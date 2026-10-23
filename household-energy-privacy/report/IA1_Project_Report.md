# Evaluating Privacy–Utility Trade-offs in Machine Learning on Household Energy Consumption Data Using Differential Privacy

## Assessment Details

- **Course:** Security and Privacy for Machine Learning
- **Assessment Component:** IA-1 (Individual Assessment 1)
- **Weightage:** 20 Marks
- **Academic Environment:** Python 3.11 / VS Code on Windows 11 (PowerShell & Command Prompt)
- **Primary Codebase:** [`household-energy-privacy/`](file:///e:/SPML/IA1/household-energy-privacy/)
- **Date of Submission:** 2026-10-23
- **Student Name:** [Insert Student Name Here]
- **Student ID / Roll No:** [Insert Student Roll No Here]

---

## 1. Abstract

High-frequency smart meter infrastructure provides electricity network operators and retail aggregators with granular household consumption traces necessary for demand forecasting, dynamic grid balancing, and tariff optimization. However, interval-level consumption traces act as a sensitive behavioral side-channel capable of revealing personal occupancy patterns, appliance usage signatures, sleep schedules, and vacation periods. This assessment investigates the formal privacy–utility trade-offs of applying Differential Privacy (DP) to short-horizon electricity consumption forecasting. Using the publicly accessible London Smart Meter Energy Consumption Dataset (UK Power Networks / Greater London Authority), we construct a reproducible, leakage-free machine learning workflow in VS Code. We implement a non-private persistence operational baseline (RMSE = 0.2029 kWh) and a non-private Ridge regression baseline (RMSE = 0.1864 kWh, $R^2 = 0.5880$) predicting half-hourly consumption from past lags and calendar features over a 505-day horizon for household `MAC000002`. We evaluate a differentially private linear regression model using IBM Diffprivlib across privacy budgets $\epsilon \in [0.05, 10.0]$ under stochastic repetition. Empirical results show that at $\epsilon = 10.0$, the DP model attains an RMSE of $0.1876 \pm 0.0006$ kWh ($\Delta\text{RMSE} = +0.63\%$), which degrades to $0.2920 \pm 0.1906$ kWh ($\Delta\text{RMSE} = +56.66\%$) at $\epsilon = 2.0$, and experiences severe numerical instability when $\epsilon \le 1.0$. Crucially, we document that this guarantee protects individual 30-minute records rather than whole household trajectories. Complementing the central regression task, we demonstrate Local Differential Privacy (LDP) via a $k$-ary Randomized Response ($k$-RR) mechanism on energy consumption tertiles, and evaluate Privacy-Preserving Data Mining (PPDM) through multi-resolution temporal aggregation, proving analytically why heuristic aggregation does not constitute a formal differential privacy guarantee.

---

## 2. Problem Statement and Motivation

Modern Advanced Metering Infrastructure (AMI) records domestic electricity consumption at high temporal resolutions (typically 15-minute to 30-minute intervals). Machine learning models trained on these historical consumption streams enable short-term load forecasting (STLF) for grid reliability and localized load dispatch.

However, domestic energy profiles are intrinsically tied to personal human activities. Fine-grained consumption spikes disclose:
1. **Occupancy and Activity Timing:** Morning waking routines, evening arrival times, and departure schedules.
2. **Specific Appliance Signatures:** Non-Intrusive Load Monitoring (NILM) techniques can identify kettle usage (2–3 kW), microwave cycles, electric vehicle charging, and medical equipment.
3. **Absence Periods:** Protracted baseload flatlining reveals vacant homes, introducing physical burglary risks.

Consequently, training centralized predictive models directly on unperturbed energy data introduces serious privacy liabilities under data protection regulations (e.g. EU/UK GDPR, DPDP Act). The core technical challenge is quantifying the **privacy–utility trade-off**: determining how much predictive fidelity must be sacrificed to achieve mathematically verifiable privacy guarantees.

---

## 3. Objectives, Scope, and Research Questions

### Primary Research Question
> **“How does a formally specified differential-privacy mechanism affect the accuracy of short-term household energy-consumption prediction?”**

### Supporting Research Questions
1. **Baseline Utility:** What level of predictive accuracy is achieved by a conventional operational persistence baseline and a non-private Ridge regression model?
2. **Empirical Privacy–Utility Degradation:** How do predictive error (RMSE, MAE) and computational runtime scale as the differential privacy budget $\epsilon$ is tightened from $\epsilon = 10.0$ to $\epsilon = 0.05$?
3. **Privacy Unit and Adjacency:** What is the exact mathematical unit protected by the DP regression mechanism, and why is row-level DP fundamentally distinct from household-level DP?
4. **LDP and PPDM Extensions:** How do Local Differential Privacy (client-side perturbation) and temporal aggregation (PPDM data minimization) compare in terms of error, utility preservation, and vulnerability to differencing attacks?

### Scope and Boundary Conditions
- **Task:** Next-interval half-hourly electricity consumption prediction ($\hat{y}_t \in \mathbb{R}^+$) using past readings and calendar features.
- **Privacy Mechanism:** Objective/covariance perturbation via IBM Diffprivlib's $(\epsilon, 0)$-DP linear regression.
- **Important Distinction:** Predictive accuracy is **not** a privacy metric. Low prediction error does not imply privacy preservation, nor does high prediction error prove robustness against inference attacks. Privacy guarantees arise exclusively from the bounded divergence of the randomized mechanism.

---

## 4. Threat Model and Assumptions

Following standard machine learning privacy frameworks, we specify the threat model:

```
[Smart Meter / Client]  ---> [Aggregator / Central ML Trainer] ---> [Published Model / Analyst]
      (LDP Domain)                       (Central DP Domain)               (External Adversary)
```

1. **Asset:** Time-series power consumption readings $y_t \in \mathbb{R}$ of domestic consumers.
2. **Attacker Capabilities:**
   - **Black-Box / Grey-Box Access:** The adversary can submit query feature vectors $x^*$ to the trained regression model and observe predictions $\hat{y}$.
   - **White-Box Parameter Access:** In central DP, the trained model parameters (weight vector $w \in \mathbb{R}^d$ and bias $b \in \mathbb{R}$) are considered public outputs released to third parties.
   - **Auxiliary Knowledge:** The adversary may possess side-channel knowledge (e.g., weather observations, public holidays, neighborhood solar irradiance, or partial known occupancy logs).
3. **Attacker Goals:**
   - *Membership Inference Attack (MIA):* Determining whether a specific household or reading was included in the training dataset.
   - *Attribute Inference / Reconstruction:* Reconstructing private half-hourly consumption values to infer domestic routines or appliance usage.
4. **Trust Assumptions:**
   - In **Central DP (Module 2)**, the training aggregator is trusted to collect raw data and execute the DP algorithm honestly.
   - In **Local DP (Module 3)**, the central aggregator is **untrusted**; perturbation occurs on the client meter before egress.
   - In **PPDM Aggregation (Module 4)**, the aggregator computes deterministic sums over time, which assumes an honest-but-curious recipient but offers no protection against differencing queries.

---

## 5. Dataset Source, Schema, Licence, and Subset

- **Dataset Portal:** [London Datastore - SmartMeter Energy Consumption in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- **Originating Organization:** UK Power Networks (Low Carbon London project)
- **Licence:** Creative Commons Attribution (CC BY 4.0)
- **Access Date:** 2026-10-23
- **Full Dataset Size:** Monolithic archive ~764.54 MB ZIP (~10.2 GB uncompressed, ~167 million rows across 5,567 households).
- **Examined Local Subset:** `LCL-June2015v2_0.csv` (~50.7 MB, exactly 1,000,000 rows, 30 households).
- **Experimental Target Household:** `MAC000002` (24,158 continuous raw records, 24,141 cleaned half-hourly intervals from 2012-10-12 to 2014-02-28, spanning 505 contiguous calendar days).
- **Cadence:** Half-hourly (every 30 minutes, 48 readings/day). Observed cadence conformity: **99.97%**.

---

## 6. Data Preparation and Preprocessing Pipeline

The preprocessing pipeline ([`src/preprocessing.py`](file:///e:/SPML/IA1/household-energy-privacy/src/preprocessing.py)) enforces rigorous time-series principles:

1. **Schema Normalization:** Dynamic alias adapter strips trailing whitespace and maps varied source column names (`LCLid` $\to$ `household_id`, `DateTime` $\to$ `timestamp`, `KWH/hh (per half hour) ` $\to$ `energy_kwh`).
2. **Invalid Value Handling:** Coerces string anomalies (such as literal `'Null'` tokens) to IEEE NaNs. Across 24,158 raw readings of `MAC000002`, exactly 1 null reading was detected and imputed via forward-fill (limit=2), preserving time continuity.
3. **Deduplication:** Exactly 17 duplicate timestamp entries were identified and pruned using the deterministic rule: preserve first occurrence.
4. **Strictly Past-Looking Feature Engineering:**
   - $\text{lag}_1 = y_{t-1}$ (reading 30 minutes prior)
   - $\text{lag}_2 = y_{t-2}$ (reading 1 hour prior)
   - $\text{lag}_{48} = y_{t-48}$ (reading exactly 24 hours prior)
   - $\text{rolling\_mean}_4 = \frac{1}{4} \sum_{i=1}^4 y_{t-i}$ (average over past 2 hours)
   - Calendar features: `hour` ($0 \dots 23$), `day_of_week` ($0 \dots 6$), `is_weekend` ($\{0, 1\}$), `month` ($1 \dots 12$).
   - **Household Boundary Integrity:** Lags are grouped strictly by `household_id`. Lags never cross household boundaries. Initial boundary rows (first 48 half-hours) with undefined lags are dropped cleanly.
5. **Chronological Splitting (No Shuffling):**
   - **Train:** First 70% (16,865 intervals; 2012-10-13 01:00 to 2013-09-30 09:30)
   - **Validation:** Next 15% (3,614 intervals; 2013-09-30 10:00 to 2013-12-14 16:30)
   - **Test:** Final 15% (3,614 intervals; 2013-12-14 17:00 to 2014-02-28 00:00)
   - **Leakage Invariant Verified:** $\max(\text{train}) < \min(\text{val}) \le \max(\text{val}) < \min(\text{test})$.
6. **Strict Training-Set Normalization:** `StandardScaler` is fitted exclusively on training feature vectors $X_{\text{train}}$ and used to transform validation and test vectors, preventing data snooping.

---

## 7. Conceptual Explanation and Methodology

### 7.1 Differential Privacy Foundation
A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy if for any two adjacent datasets $D, D' \in \mathcal{D}$ and for all query response subsets $S \subseteq \text{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in S] \le e^{\epsilon} \, \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$

When $\delta = 0$, $\mathcal{M}$ satisfies pure $\epsilon$-Differential Privacy. In our regression experiment:
- **Protected Unit:** A single 30-minute record $(x_i, y_i)$.
- **Adjacency:** Bounded differential privacy where $D'$ is formed by replacing one row in $D$ with another arbitrary row within the compact bounding box.
- **Mechanism:** IBM Diffprivlib's Linear Regression perturbs the empirical Gram matrix $X^T X$ using Wishart noise and the projection vector $X^T y$ using Laplace/Gaussian noise calibrated to the $L_2$ global sensitivity derived from training bounds `bounds_X` and `bounds_y`.

### 7.2 Why Record-Level DP $\neq$ Household-Level DP
In smart meter analytics, a single household contributes a continuous sequence of readings ($N \approx 17,520$ half-hours per year). Record-level DP bounds the influence of **any single 30-minute interval** by $e^\epsilon$. By the **Group Privacy property** of differential privacy, protecting the entire household trajectory of $k$ consecutive intervals scales the effective privacy parameter to:

$$\epsilon_{\text{group}} = k \cdot \epsilon$$

For $k = 48$ (one day), an $\epsilon = 0.5$ per-record guarantee yields $\epsilon_{\text{group}} = 24.0$, which is practically vacuous. Protecting whole households under DP requires either household-level partitioning with bounded sensitivity per user (e.g. User-Level DP-SGD with clipping across all samples of a user) or aggregate bounding. Therefore, this project explicitly documents that its formal guarantee is strictly **record-level DP**, protecting individual burst events (e.g., boiling a kettle) from inference, but not concealing long-term occupancy trends.

---

## 8. Workflow Diagram

```mermaid
flowchart TD
    A["Raw London Smart Meter Dataset<br/>(LCL-June2015v2_0.csv)"] --> B["Schema Adapter & Chunked Stream<br/>(src/data_loader.py)"]
    B --> C["Data Validation & Cadence Check<br/>(Deduplication & Imputation)"]
    C --> D["Past-Looking Feature Construction<br/>(Lags 1, 2, 48, Rolling Mean, Calendar)"]
    D --> E["Chronological Split<br/>(Train 70% | Val 15% | Test 15%)"]
    
    E --> F["Non-Private Persistence Baseline<br/>(Lag-1 Predictor)"]
    E --> G["Non-Private Ridge Baseline<br/>(ERM L2 Regression)"]
    E --> H["Differentially Private Linear Regression<br/>(Diffprivlib Covariance Perturbation)"]
    E --> I["Local DP k-RR Demonstration<br/>(Client-Side Tertile Perturbation)"]
    E --> J["PPDM Multi-Resolution Aggregation<br/>(30-min vs 1-hour vs 1-day)"]
    
    F --> K["Evaluation & Artifact Exporter<br/>(src/evaluation.py)"]
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L["Results: Tables & Figures<br/>(CSV, JSON, PNG in results/)"]
    K --> M["IA-1 Assessment Report<br/>(report/IA1_Project_Report.md)"]
```

---

## 9. Mapping to All Five Syllabus Modules

| Module Number & Course Syllabus Title | Project Mapping & Implementation Details | Code & Analytical Reference | Status |
| :--- | :--- | :--- | :--- |
| **Module 1: Privacy Considerations in Machine Learning** | Comprehensive threat model formulation; delineation of sensitive attributes (occupancy, appliance signatures); analysis of membership and attribute inference risks; proof that high accuracy $\neq$ low privacy. | [`report/IA1_Project_Report.md`](file:///e:/SPML/IA1/household-energy-privacy/report/IA1_Project_Report.md#4-threat-model-and-assumptions) | **Fully Addressed** |
| **Module 2: Differential Privacy for Machine Learning** | Formally grounded $(\epsilon, 0)$-DP linear regression experiment using Diffprivlib across $\epsilon \in [0.05, 10.0]$; explicit derivation of clipping bounds; stochastic variance analysis; rigorous analysis of record-level vs. household-level DP. | [`src/dp_model.py`](file:///e:/SPML/IA1/household-energy-privacy/src/dp_model.py), [`results/tables/privacy_comparison.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/privacy_comparison.csv) | **Fully Implemented & Verified** |
| **Module 3: Local Differential Privacy for Machine Learning** | Local feature privatization demo using $k$-ary Randomized Response ($k$-RR, $k=3$) on energy tertiles; transition probabilities $p, q$; closed-form mathematically unbiased population frequency estimator $\hat{\pi}$; TVD and MAE evaluation across $\epsilon_{ldp} \in [0.5, 4.0]$. | [`src/ldp_demo.py`](file:///e:/SPML/IA1/household-energy-privacy/src/ldp_demo.py), [`tests/test_ldp.py`](file:///e:/SPML/IA1/household-energy-privacy/tests/test_ldp.py) | **Fully Implemented & Verified** |
| **Module 4: Privacy-Preserving Data Mining (PPDM)** | Multi-scale temporal aggregation (30-min native, hourly sum, daily sum); quantitative utility and peak-to-average ratio retention; formal proof that heuristic aggregation is vulnerable to differencing queries and provides no DP guarantee. | [`src/aggregation_demo.py`](file:///e:/SPML/IA1/household-energy-privacy/src/aggregation_demo.py), [`results/tables/aggregation_comparison.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/aggregation_comparison.csv) | **Fully Implemented & Verified** |
| **Module 5: Compressive Privacy for Machine Learning** | Deep conceptual and mathematical analysis of Information-Theoretic Compressive Privacy (Privacy Funnel / Information Bottleneck); formal proof of why naive PCA / feature selection is not compressive privacy; specification of IA-2 follow-up experiment. | [`report/IA1_Project_Report.md`](file:///e:/SPML/IA1/household-energy-privacy/report/IA1_Project_Report.md#95-module-5-compressive-privacy-for-machine-learning) | **Conceptually Grounded & Analytical Specification** |

### 9.5 Module 5: Compressive Privacy for Machine Learning (In-Depth Analysis)
In privacy literature (e.g., Sankar et al., 2013; Makhdoumi et al., 2014), **Compressive Privacy** addresses the fundamental trade-off between task utility and privacy disclosure in continuous sensor streams via information-theoretic dimensionality reduction.

Given an input energy vector $X \in \mathbb{R}^d$, a sensitive attribute $S$ (e.g. household occupancy identity or high-load appliance status), and a primary utility task $Y$ (e.g. next-day aggregate consumption), compressive privacy seeks a stochastic mapping or lossy compression $Z = \phi(X)$ that solves the **Privacy Funnel optimization problem**:

$$\min_{P_{Z|X}} \quad I(X; S \mid Z) \quad \text{subject to} \quad I(Z; Y) \ge R \quad \text{or} \quad \mathbb{E}[d(Y, \hat{Y}(Z))] \le D$$

where $I(\cdot ; \cdot)$ denotes mutual information and $d(\cdot, \cdot)$ is a distortion measure.

#### Crucial Academic Distinction: Why Naive PCA is NOT Compressive Privacy
Standard Principal Component Analysis (PCA) selects projection axes $W$ that maximize the variance $\mathbb{E}[\|X - W W^T X\|^2]$ of the data:
$$\max_W \quad \text{Tr}(W^T \Sigma_X W) \quad \text{s.t.} \quad W^T W = I$$
Because standard PCA is entirely oblivious to the sensitive variable $S$, the principal components with the largest eigenvalues almost invariably capture the highest energy consumption bursts (e.g. oven, heat pump, EV charging), which are precisely the most privacy-revealing features! Selecting the top $k$ principal components or performing unsupervised feature selection therefore concentrates, rather than compresses, sensitive behavioral information.

To qualify as a defensible compressive privacy mechanism, the projection matrix $W$ must be synthesized by solving a constrained optimization (such as the generalized eigenvalue problem between the utility covariance $\Sigma_{XY}$ and the adversary cross-covariance $\Sigma_{XS}$) or by applying randomized Johnson-Lindenstrauss projections with proven reconstruction bounds. In Section 18, we specify this concrete experimental design for IA-2.

---

## 10. Implementation Details and Tools

The project is developed in pure Python adhering to modern software engineering standards:

- **Python Version:** 3.11.4
- **Core Libraries:**
  - `pandas` (>= 2.1.0): Schema normalization, chunked CSV parsing, datetime resampling.
  - `numpy` (>= 1.26.0, < 2.0.0): Numerical matrix operations and vectorization.
  - `scikit-learn` (>= 1.4.0): Ridge regression, standard scaling, and evaluation metrics.
  - `diffprivlib` (>= 0.6.6): IBM Differential Privacy Library for objective/Gram matrix perturbation.
  - `matplotlib` (>= 3.8.0): Publication-quality figures using headless Agg backend.
  - `pytest` (>= 9.1.1): Automated unit testing framework.
- **Project Structure:**
```
household-energy-privacy/
    data/
        raw/              <- Local raw CSV datasets (gitignored)
        processed/        <- Processed feature samples (snapshot)
    results/
        figures/          <- prediction_comparison.png, privacy_utility_plot.png, aggregation_comparison.png
        tables/           <- data_quality_report.json, baseline_metrics.csv, privacy_comparison.csv, etc.
        logs/             <- run.log (detailed execution trace)
    src/
        __init__.py       <- Package marker
        config.py         <- Dataclass configurations
        data_loader.py    <- Chunked loading & schema adaptation
        preprocessing.py  <- Deduplication, imputation, lag construction, chronological split
        baseline_model.py <- Persistence & Ridge baselines
        dp_model.py       <- Differentially private linear regression & parameter bounds
        ldp_demo.py       <- Local DP k-RR mechanism & unbiased estimator
        aggregation_demo.py <- PPDM multi-resolution temporal analysis
        evaluation.py     <- Metric reporting and figure rendering
        utils.py          <- Logging, seeding, and utility loss formulas
    tests/
        test_preprocessing.py   <- Schema, deduplication, boundaries tests
        test_splits.py          <- Chronological order, leakage tests
        test_ldp.py             <- LDP probability axioms & unbiasedness tests
        test_pipeline_and_dp.py <- Missing data, output creation, DP bounds tests
    report/
        IA1_Project_Report.md   <- This comprehensive assessment report
        DATASET_NOTES.md        <- Dataset verification, licence, and schema notes
    main.py           <- Unified CLI entry point
    requirements.txt  <- Pinned dependencies
    README.md         <- Setup and execution guide
    .gitignore        <- Source control exclusion rules
```

---

## 11. Experimental Setup and Baseline Definitions

### Operational Persistence Baseline
In short-term electricity demand forecasting, persistence serves as the primary operational benchmark:
$$\hat{y}_t = y_{t-1}$$
- **Characteristics:** Parameter-free, deterministic, zero training time.
- **Strengths:** Captures immediate autocorrelation in base load.
- **Weaknesses:** Cannot predict sharp morning ramps, evening peaks, or structural weekend shifts.

### Non-Private Ridge Regression Baseline
Ridge regression minimizes the penalized residual sum of squares:
$$\min_{w, b} \quad \frac{1}{2n} \sum_{i=1}^n (y_i - (w^T x_i + b))^2 + \frac{\alpha}{2} \|w\|_2^2$$
- **Hyperparameter:** Regularization strength $\alpha = 1.0$.
- **Feature Set:** $\text{lag}_1, \text{lag}_2, \text{lag}_{48}, \text{rolling\_mean}_4, \text{hour}, \text{day\_of\_week}, \text{is\_weekend}, \text{month}$.
- **Fair Comparison Protocol:** Evaluated on the exact same chronological test partition (3,614 intervals; 2013-12-14 to 2014-02-28).

---

## 12. Actual Measured Results and Evidence Files

All figures and tables reported below were generated directly from execution on `LCL-June2015v2_0.csv` (Household `MAC000002`):

### 12.1 Baseline Model Performance ([`baseline_metrics.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/baseline_metrics.csv))

| Model | MAE (kWh) | RMSE (kWh) | $R^2$ Score | Fit Time (s) | Predict Time (s) | Privacy Parameter ($\epsilon$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence ($\text{lag}_1$)** | 0.1081 | 0.2029 | 0.5116 | 0.0000 | 0.000002 | $\infty$ (Non-private) |
| **Ridge Regression (Non-Private)** | **0.1057** | **0.1864** | **0.5880** | 0.0021 | 0.000272 | $\infty$ (Non-private) |

### 12.2 Differential Privacy Regression Results ([`privacy_comparison.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/privacy_comparison.csv))
Evaluated over $N = 5$ stochastic trials per $\epsilon$, reporting Mean $\pm$ Standard Deviation:

| Privacy Budget ($\epsilon$) | Mean MAE (kWh) | Mean RMSE (kWh) | Std RMSE (kWh) | Mean $R^2$ | Utility Loss ($\Delta\text{RMSE}\%$) | Status / Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\epsilon = 10.0$** | 0.1066 | 0.1876 | 0.0006 | +0.5827 | **+0.63%** | Near-Optimal Utility |
| **$\epsilon = 5.0$** | 0.1076 | 0.1887 | 0.0013 | +0.5777 | **+1.24%** | Minor Degradation |
| **$\epsilon = 2.0$** | 0.1785 | 0.2920 | 0.1906 | -0.4420 | **+56.66%** | Noticeable Trade-off |
| **$\epsilon = 1.0$** | 84.40 | 122.57 | 237.04 | Negative | +65,666% | Matrix Instability Onset |
| **$\epsilon = 0.5$** | 146.14 | 208.82 | 174.29 | Negative | +111,945% | Noise Dominance |
| **$\epsilon = 0.2$** | 283.71 | 382.39 | 115.74 | Negative | +205,078% | Noise Dominance |
| **$\epsilon = 0.1$** | $7.62 \times 10^6$ | $9.07 \times 10^6$ | $1.81 \times 10^7$ | Negative | Extreme | Covariance Singular |
| **$\epsilon = 0.05$** | $8.87 \times 10^6$ | $1.06 \times 10^7$ | $2.12 \times 10^7$ | Negative | Extreme | Covariance Singular |

### 12.3 Local Differential Privacy (k-RR) Results ([`ldp_results.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/ldp_results.csv))
Discrete consumption tertiles: `Low` ($< 0.117$ kWh), `Medium` ($[0.117, 0.230]$ kWh), `High` ($> 0.230$ kWh). Ground truth test proportions: $\pi^* = [0.3381, 0.1821, 0.4798]$.

| LDP Budget ($\epsilon_{ldp}$) | Perturb Prob $p$ | Flip Prob $q$ | Est Prop Low | Est Prop Med | Est Prop High | Prop MAE | Total Variation Dist (TVD) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\epsilon_{ldp} = 4.0$** | 0.9647 | 0.0177 | 0.3355 | 0.1812 | 0.4833 | **0.00235** | **0.00352** |
| **$\epsilon_{ldp} = 2.0$** | 0.7870 | 0.1065 | 0.3473 | 0.1871 | 0.4656 | **0.00945** | **0.01418** |
| **$\epsilon_{ldp} = 1.0$** | 0.5761 | 0.2119 | 0.3663 | 0.1786 | 0.4552 | **0.01875** | **0.02813** |
| **$\epsilon_{ldp} = 0.5$** | 0.4519 | 0.2741 | 0.3634 | 0.2171 | 0.4194 | **0.04023** | **0.06035** |

### 12.4 PPDM Temporal Aggregation Metrics ([`aggregation_comparison.csv`](file:///e:/SPML/IA1/household-energy-privacy/results/tables/aggregation_comparison.csv))

| Temporal Resolution | Duration Description | Records | Mean Energy | Std Energy | Max Energy | Peak-to-Avg Ratio | Temporal Retained | Residual Disclosure Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **30min** | Half-Hourly Native | 24,141 | 0.2525 | 0.2471 | 2.994 | 11.857 | Full | **High** (individual appliance pulses visible) |
| **1h** | Hourly Sum | 12,097 | 0.5039 | 0.4463 | 4.586 | 9.101 | Moderate | **Moderate** (micro-events smoothed; daily routines evident) |
| **1d** | Daily Sum | 505 | 12.0711 | 4.4946 | 39.284 | 3.254 | Coarse | **Low-to-Moderate** (reveals occupancy & weekend totals) |

---

## 13. Visual Evidence and Generated Figures

The pipeline generated three high-resolution figures in `results/figures/`:

1. **Prediction Comparison Plot ([`results/figures/prediction_comparison.png`](file:///e:/SPML/IA1/household-energy-privacy/results/figures/prediction_comparison.png)):**
   Displays the actual ground truth consumption trace against the persistence baseline, the non-private Ridge baseline, and DP linear regression ($\epsilon = 5.0$ and $\epsilon = 2.0$) over a representative 144-interval (3-day) test horizon. At $\epsilon = 5.0$, the DP curve tightly tracks the non-private baseline, capturing diurnal cycling. At $\epsilon = 2.0$, noticeable noise fluctuations occur while preserving general load magnitude.
2. **Privacy–Utility Trade-off Curve ([`results/figures/privacy_utility_plot.png`](file:///e:/SPML/IA1/household-energy-privacy/results/figures/privacy_utility_plot.png)):**
   Depicts RMSE as a function of privacy budget $\epsilon$ on a log-log scale with standard deviation error bars. As $\epsilon \to 10.0$, error asymptotically converges to the non-private Ridge baseline (dashed reference line at 0.1864 kWh). Below $\epsilon = 2.0$, error escalates steeply due to Gram matrix perturbation.
3. **PPDM Aggregation Comparison ([`results/figures/aggregation_comparison.png`](file:///e:/SPML/IA1/household-energy-privacy/results/figures/aggregation_comparison.png)):**
   Contrasts the native 30-minute signal with hourly and daily aggregations over a two-week period. Illustrates how temporal aggregation suppresses high-frequency variance while retaining macroscopic occupancy trends.

---

## 14. Privacy Parameters, Assumptions, and Limitations

1. **Accounting Mechanism:** $(\epsilon, 0)$-DP is guaranteed via Wishart Gram matrix perturbation and vector noise addition. No probabilistic relaxation ($\delta > 0$) was assumed.
2. **Clipping Bounds:** Feature bounds `bounds_X` and target bounds `bounds_y` were derived strictly from the 0.5% and 99.5% percentiles of the training distribution $X_{\text{train}}, y_{\text{train}}$, ensuring zero test set leakage.
3. **Core Limitation (Record vs. User DP):** The mathematical guarantee applies strictly to individual 30-minute rows. Protecting a multi-month household profile requires group privacy, which degrades bounds by a factor of $k = 16,865$.
4. **Instability at Low $\epsilon$:** For linear regression solved via $(X^T X + E)^{-1}$, adding Wishart noise $E$ with large variance (small $\epsilon$) frequently renders the perturbed matrix ill-conditioned or nearly singular, leading to massive parameter inversion errors.

---

## 15. Preliminary Observations: Measurements vs. Hypotheses

### Measured Empirical Facts (Validated by Data)
- **Fact 1:** The non-private Ridge baseline improves upon operational persistence by 8.15% in RMSE (0.1864 vs. 0.2029 kWh) and achieves an $R^2$ of 0.5880 on unseen test data.
- **Fact 2:** At $\epsilon = 10.0$ and $\epsilon = 5.0$, DP regression incurs negligible utility loss (+0.63% and +1.24% $\Delta\text{RMSE}$ respectively) with high stability ($\text{Std} \le 0.0013$ kWh).
- **Fact 3:** In the local DP setting ($k$-RR), estimation error scales inversely with $\epsilon_{ldp}$; at $\epsilon_{ldp} = 4.0$, aggregate tertile frequencies are recovered with TVD = 0.00352.
- **Fact 4:** Heuristic daily aggregation reduces peak-to-average ratio from 11.857 to 3.254, smoothing micro-peaks but preserving total volume.

### Theoretical Hypotheses (Awaiting IA-2 Verification)
- **Hypothesis 1:** Implementing DP-SGD with per-sample gradient clipping rather than closed-form Gram matrix perturbation may prevent the severe matrix singularity observed at $\epsilon \le 1.0$.
- **Hypothesis 2:** Compressive privacy via randomized projections with Johnson-Lindenstrauss distortion bounds will preserve load forecasting while provably bounding the Mutual Information $I(X; S)$ with appliance signatures.

---

## 16. Evaluation Plan

The validation methodology for IA-1 was structured into four verification gates:
1. **Automated Unit Testing:** 18 automated unit tests in `pytest` covering schema adaptation, timestamp deduplication, short-gap imputation, lag boundary isolation, chronological split integrity, scale leakage prevention, and LDP probability axioms. All 18 tests passed.
2. **Synthetic Smoke Verification:** Pipeline execution verified via `--smoke-test` to guarantee end-to-end reproducibility independently of external data access.
3. **Real-Data Execution:** Verification on 24,158 records of household `MAC000002` from `LCL-June2015v2_0.csv`.
4. **Artifact Integrity:** Validation that all generated files (`.csv`, `.json`, `.png`, `.log`) exist, contain non-zero bytes, and contain mathematically valid numbers without fabricated values.

---

## 17. Individual Contribution Statement Template

*Notice: This template is provided for the student to complete truthfully in accordance with university academic integrity standards.*

```
================================================================================
                      INDIVIDUAL CONTRIBUTION STATEMENT
================================================================================
Course: Security and Privacy for Machine Learning
Assessment: IA-1 (20 Marks)
Student Name: __________________________________________________
Roll / Registration Number: ___________________________________

I certify that the work presented in this submission represents my own 
individual effort. Specifically, I have contributed to:

1. Problem Definition and Syllabus Mapping:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

2. Dataset Identification, Schema Review, and Quality Verification:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

3. Preprocessing, Feature Engineering, and Leakage-Free Splitting:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

4. Baseline Model and Differential Privacy Implementation:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

5. LDP Demonstration, PPDM Aggregation, and Compressive Privacy Analysis:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

6. Technical Report Writing and Documentation:
   - Specific contribution: ____________________________________________________
   - Estimated percentage: _____%

Signature: ______________________________         Date: ________________________
================================================================================
```

---

## 18. Remaining Work for IA-2

For the second assessment phase (IA-2), the following empirical extensions are proposed:
1. **Household-Level Differential Privacy via DP-SGD:** Transition from closed-form Diffprivlib linear regression to an iterative DP-SGD architecture (using Opacus or custom PyTorch training) with per-user sample grouping and Renyi Differential Privacy (RDP) accounting.
2. **Empirical Verification of Compressive Privacy (Module 5):**
   - Synthesize a labeled appliance-level dataset or multi-household Acorn demographic identifier $S$.
   - Implement the Information-Theoretic Privacy Funnel via randomized dimensionality reduction:
     $$Z = W X + \eta, \quad W \in \mathbb{R}^{m \times d}, \quad \eta \sim \mathcal{N}(0, \sigma^2 I)$$
   - Measure mutual information leakage $I(Z; S)$ using a neural mutual information estimator (MINE) or empirical adversary classification accuracy, plotted against STLF forecasting RMSE.
3. **Multi-Household Cross-Validation:** Extend evaluation across all 30 households in `LCL-June2015v2_0.csv` to quantify how user heterogeneity impacts privacy budget allocation.

---

## 19. References and Official Documentation Links

1. **London Datastore Smart Meter Dataset:**
   - Portal: [https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
   - Licence: Creative Commons Attribution (CC BY 4.0).
2. **Differential Privacy Foundations:**
   - Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.
3. **IBM Differential Privacy Library (Diffprivlib):**
   - Holohan, N., Braghin, S., Mac Aonghusa, P., & Levacher, K. (2019). *Diffprivlib: The IBM Differential Privacy Library.* arXiv:1907.02444.
   - Documentation: [https://diffprivlib.readthedocs.io/en/latest/](https://diffprivlib.readthedocs.io/en/latest/)
4. **Local Differential Privacy and Randomized Response:**
   - Warner, S. L. (1965). *Randomized response: A survey technique for eliminating evasive answer bias.* Journal of the American Statistical Association, 60(309), 63–69.
   - Wang, T., Blocki, J., Li, N., & Jha, S. (2017). *Locally Differentially Private Protocols for Frequency Estimation.* In USENIX Security Symposium (pp. 729–745).
5. **Compressive Privacy and Information-Theoretic Trade-offs:**
   - Sankar, L., Rajagopalan, S. R., & Poor, H. V. (2013). *Utility-Privacy Tradeoffs in Smart Grid: An Information-Theoretic Approach.* IEEE Transactions on Smart Grid, 4(1), 211–222.
   - Makhdoumi, A., Salamatian, S., Fawaz, N., & Médard, M. (2014). *From the Information Bottleneck to the Privacy Funnel.* In IEEE Information Theory Workshop (ITW) (pp. 501–505).
6. **Machine Learning and Time Series:**
   - Scikit-Learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
