# London Smart Meter Energy Consumption Dataset: Metadata and Access Notes

## 1. Official Dataset Identification

- **Dataset Title:** SmartMeter Energy Consumption Data in London Households
- **Official Portal URL:** [https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- **Originating Project:** Low Carbon London (LCL) project led by UK Power Networks
- **Publishing Authority:** Greater London Authority (GLA) / London Datastore
- **Licence:** Creative Commons Attribution (CC BY 4.0) / UK Open Government Licence (OGL v3) compatible
- **Date Verified & Accessed:** 2026-10-23
- **Temporal Coverage of Full Project:** November 2011 – February 2014

---

## 2. Portal Availability and File Structure

The official London Datastore portal publishes the half-hourly smart meter readings in two alternative packaging formats:

1. **Single Monolithic Archive:**
   - Compressed ZIP size: **~764.54 MB**
   - Uncompressed size: **~10.2 GB**
   - Record count: **~167 million rows** spanning 5,567 households.

2. **Split Archive (168 CSV files):**
   - Compressed ZIP size: **~758.86 MB**
   - Partitioning: 168 separate CSV files (`LCL-June2015v2_0.csv` through `LCL-June2015v2_167.csv`), each containing roughly 1,000,000 half-hourly readings.
   - Purpose: Facilitates memory-conscious inspection without downloading or decompressing the full 10 GB monolithic file.

---

## 3. Investigated Local Subset Metadata

For this IA-1 assessment, the local subset file `LCL-June2015v2_0.csv` is utilized directly:

- **Local File Path:** `e:\SPML\IA1\LCL-June2015v2_0.csv`
- **Exact File Size:** `50,755,532 bytes` (~50.7 MB)
- **Total Record Count:** Exactly `1,000,000` rows of half-hourly consumption readings.
- **Household Count:** Contains 30 unique households (`MAC000002` through `MAC000034`).
- **Focus Household (`MAC000002`):**
  - Raw reading count: 24,158 rows.
  - Cleaned & deduplicated reading count: 24,141 rows.
  - Date span: 2012-10-12 00:30:00 to 2014-02-28 00:00:00 (~505 contiguous calendar days).
  - Interval cadence: Half-hourly (every 30 minutes; 48 readings per calendar day).
  - Cadence conformity: 99.97% (only 6 missing timestamps over 505 days).
  - Consumption statistics: Min = 0.000 kWh, Median = 0.158 kWh, Mean = 0.252 kWh, Max = 2.994 kWh.

---

## 4. Schema and Data Dictionary

Inspection of the raw CSV header reveals:
```csv
LCLid,stdorToU,DateTime,KWH/hh (per half hour) 
```

| Source Column Header | Normalized Field Name | Data Type | Description | Observed Artifacts & Notes |
| :--- | :--- | :--- | :--- | :--- |
| `LCLid` | `household_id` | String | Unique anonymized household meter identifier | E.g. `MAC000002`, `MAC000018`. |
| `stdorToU` | `tariff_type` | Categorical | Tariff scheme applied to the customer | `Std` (Standard tariff) or `ToU` (Time of Use dynamic pricing). |
| `DateTime` | `timestamp` | Datetime (ISO) | Half-hourly interval start timestamp | Format: `YYYY-MM-DD HH:MM:SS.0000000`. |
| `KWH/hh (per half hour) ` | `energy_kwh` | Float64 | Electricity consumed in kilowatt-hours per 30 minutes | Trailing whitespace in header; contains string `'Null'` for missing sensor readings. |

### Schema Adapter Requirements:
1. Strip leading and trailing whitespace from column headers (e.g. `'KWH/hh (per half hour) '` -> `'KWH/hh (per half hour)'`).
2. Map canonical aliases dynamically (`timestamp`, `household_id`, `energy_kwh`).
3. Parse numeric values using `pd.to_numeric(errors='coerce')` so that string tokens such as `'Null'` are converted to standard IEEE floating-point NaNs rather than forcing `object` dtypes.
4. Detect duplicate `(household_id, timestamp)` pairs and apply the deterministic rule: keep first occurrence.

---

## 5. Memory-Conscious Loading and Disk Safety

1. **No Automatic Unchecked Downloads:** The project adheres to strict disk safety policies. The full dataset (~10 GB uncompressed) is never automatically downloaded.
2. **Chunked Streaming:** `src/data_loader.py` reads datasets in bounded blocks (default `chunk_size = 50,000` rows) using Python generators, filtering target households on-the-fly to guarantee minimal RAM footprint (<150 MB peak RAM).
3. **Source Control Protection:** `.gitignore` explicitly excludes `*.csv`, `data/raw/*.csv`, and `data/processed/*.csv`. Only code, configuration, reports, and generated small artifact summaries are committed.
