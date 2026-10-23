# Evaluating Privacy–Utility Trade-offs in Machine Learning on Household Energy Consumption Data Using Differential Privacy

An individual college assessment project for **Security and Privacy for Machine Learning** (IA-1, 20 Marks).

---

## 1. Project Purpose

This project evaluates the empirical and mathematical trade-offs between predictive accuracy (utility) and formal privacy guarantees when training machine learning models on high-frequency domestic smart meter electricity data.

Key capabilities implemented in ordinary, modular Python files:
- **Time-Series Forecasting:** Predicts next-interval half-hourly electricity consumption from past lags and calendar features without lookahead leakage.
- **Operational & ML Baselines:** Persistence predictor and Scikit-Learn Ridge regression.
- **Differential Privacy Regression:** Formally grounded $(\epsilon, 0)$-DP linear regression via IBM Diffprivlib across privacy budgets $\epsilon \in [0.05, 10.0]$ with stochastic variance analysis.
- **Local Differential Privacy (LDP):** Client-side $k$-ary Randomized Response ($k$-RR) mechanism on consumption tertiles with an unbiased aggregate estimator.
- **Privacy-Preserving Data Mining (PPDM):** Multi-resolution temporal aggregation (30-min vs 1-hour vs 1-day) demonstrating why heuristic aggregation is not a formal DP guarantee.
- **Compressive Privacy Analysis:** Detailed information-theoretic framing of the Privacy Funnel and proposed IA-2 experiment.

---

## 2. Prerequisites

- **Operating System:** Windows 10/11 (x64)
- **Python:** Python 3.10 or 3.11 (tested on Python 3.11.4)
- **Editor:** Visual Studio Code (VS Code) with Python extension

---

## 3. Environment Setup

### Option A: Windows PowerShell Setup (Standard)

Open a PowerShell terminal in the project directory:

```powershell
# Navigate into the project folder
cd e:\SPML\IA1\household-energy-privacy

# Create a dedicated virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install pinned dependencies
pip install -r requirements.txt
```

> **PowerShell Execution Policy Troubleshooting:**  
> If PowerShell blocks script activation with an execution policy error (`PSSecurityException`), do **not** disable system-wide security settings. Instead, bypass the restriction for the current process only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

---

### Option B: Windows Command Prompt (cmd.exe) Alternative

If PowerShell activation is restricted by your system administrator:

```cmd
cd e:\SPML\IA1\household-energy-privacy
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Setting the VS Code Python Interpreter

1. Open VS Code in the workspace folder (`File` $\to$ `Open Folder...` $\to$ `e:\SPML\IA1\household-energy-privacy`).
2. Press `Ctrl + Shift + P` to open the Command Palette.
3. Type `Python: Select Interpreter` and press `Enter`.
4. Choose the local virtual environment interpreter:  
   `.\.venv\Scripts\python.exe` (or `Python 3.11 ('.venv': venv)`).

---

## 4. Dataset Acquisition and Placement

This project uses the official **SmartMeter Energy Consumption Data in London Households** dataset published by UK Power Networks and Greater London Authority on the London Datastore.

- **Official Portal:** [https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- **Licence:** Creative Commons Attribution (CC BY 4.0)

### Manual Placement
1. Download a split archive (e.g., `LCL-June2015v2_0.csv`).
2. Place the CSV file in either:
   - `data/raw/LCL-June2015v2_0.csv`, OR
   - The parent folder `e:\SPML\IA1\LCL-June2015v2_0.csv`.
3. The project data loader automatically detects the file across these standard search locations.

*(Note: Raw data is strictly excluded from Git tracking via `.gitignore`.)*

---

## 5. Running the Pipeline

### Full Experiment (Real Dataset)
Run the complete pipeline on household `MAC000002`:

```powershell
python main.py
```

### Optional Command-Line Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-path` | string | Auto-detect | Explicit path to a local CSV file |
| `--household-id` | string | `MAC000002` | Anonymized household meter ID to analyze |
| `--subset-size` | int | `25000` | Maximum rows to stream from the raw CSV |
| `--smoke-test` | flag | False | Runs synthetic smoke test (no real data required) |
| `--skip-plots` | flag | False | Skips generating matplotlib figure PNGs |
| `--epsilons` | float list | `0.05 0.1 0.2 ...` | Privacy budgets $\epsilon$ to evaluate |
| `--n-repeats` | int | `5` | Stochastic trials per epsilon for error bounds |

**Examples:**
```powershell
# Run with custom epsilon values and 10 repeats
python main.py --epsilons 0.5 1.0 2.0 5.0 --n-repeats 10

# Run in quick smoke-test mode using synthetic data
python main.py --smoke-test
```

---

## 6. Running Unit Tests

All unit tests are executed using `pytest`:

```powershell
# Run all unit tests with verbose output
python -m pytest -v

# Run a specific test suite
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_ldp.py -v
python -m pytest tests/test_splits.py -v
python -m pytest tests/test_pipeline_and_dp.py -v
```

---

## 7. Output Artifacts and Results

All generated outputs are systematically organized in the `results/` folder:

```
results/
├── figures/
│   ├── prediction_comparison.png     <- Actual vs Persistence vs Ridge vs DP predictions
│   ├── privacy_utility_plot.png      <- Privacy budget epsilon vs RMSE curve with error bars
│   └── aggregation_comparison.png    <- Multi-resolution temporal profile comparison
├── tables/
│   ├── data_quality_report.json      <- Data cleaning, duplicate counts, cadence conformity
│   ├── baseline_metrics.csv          <- MAE, RMSE, R2, and runtime for non-private baselines
│   ├── privacy_comparison.csv        <- Mean and std metrics across all epsilon values
│   ├── ldp_results.csv               <- k-RR estimation error, TVD, and true/est proportions
│   ├── aggregation_comparison.csv    <- 30-min vs 1-hour vs 1-day descriptive statistics
│   └── run_config.json               <- Exact parameters and metadata of the experiment run
└── logs/
    └── run.log                       <- Formatted timestamped execution log
```

---

## 8. Ethical, Legal, and Privacy Limitations

1. **Lawful Data Use:** This work uses exclusively public open-data published under CC BY 4.0. No private smart meter records, consumer identities, or passwords are used or stored.
2. **Record-Level vs. Household-Level DP:** The implemented Differential Privacy Linear Regression protects **individual 30-minute interval observations**, not the entire multi-year household time series. By the group privacy property, protecting $k = 16,865$ intervals incurs a multiplicative budget scaling $k \cdot \epsilon$.
3. **Utility is Not a Privacy Metric:** Predictive accuracy alone is not a measure of privacy. A model with high prediction error may still leak sensitive membership information, while a model with strong DP guarantees may still forecast aggregate patterns accurately.
4. **Heuristic Aggregation Caveat:** Temporal aggregation (PPDM) reduces high-frequency variance but offers **zero formal differential privacy**. Differencing queries over overlapping intervals completely reconstruct intermediate readings unless randomized noise is added.
