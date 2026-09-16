# Retail Command Center: Apparel Pre-Sales & RIM Valuation Forensics

Enterprise-grade Retail Inventory Method (RIM) balance sheet simulator, inventory valuation engine, and store performance anomaly forensics.

## Overview

This repository provides a 50-store, 24-month rolling monthly apparel dataset (1,200 observations) engineered for:
1. **Retail Inventory Method (RIM) Accounting**: Cumulative markon complement, shrinkage, markdowns, and ending inventory cost valuation.
2. **Multi-Period Balance Sheet Integrity**: Strict roll-forward continuity ($T+1$ Beginning Inventory = $T$ Ending Inventory).
3. **Store Anomaly Forensics**: Injected operational patterns for clustering and machine learning discovery (Discount-Addicted, Shrinkage Leakage, High-Turn Flagships).

## Setup & Execution

### 1. Environment Setup
```powershell
# Create isolated virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
```powershell
python generate_apparel_data.py
```

This generates:
- `synthetic_apparel_rim_data.csv`
- `synthetic_apparel_rim_data.parquet`
- Console audit summary with RIM balance sheet verification and anomaly diagnostics.
