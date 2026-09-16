"""
Enterprise Apparel Retail Inventory Method (RIM) Dataset Generator.

Generates a 50-store, 24-month rolling monthly apparel dataset (1,200 rows)
structured for Retail Inventory Method (RIM) balance-sheet accounting,
inventory valuation forensics, and operational anomaly detection.

Technical Specifications:
- Reproducible deterministic generation via NumPy random seed.
- Strict multi-period temporal integrity (Month T ending balance == Month T+1 beginning balance).
- RIM accounting: Cost-to-Retail complement, reductions (Sales, Markdowns, Shrinkage),
  and ending inventory cost valuation.
- Injected operational anomalies:
  * Discount-Addicted Stores (>35% markdown-to-sales ratio)
  * Shrinkage Anomaly Stores (4.5% - 6.5% inventory shrinkage)
  * High-Turn / Capital-Efficient Flagships (Lean inventory, low markdowns, top-tier GMROI)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def set_reproducible_seed(seed: int = 42) -> np.random.Generator:
    """Initialize a dedicated NumPy random generator for full reproducibility."""
    return np.random.default_rng(seed)


def define_store_network(rng: np.random.Generator) -> pd.DataFrame:
    """
    Construct the 50-store metadata network partitioned across regions, tiers, and operational profiles.
    
    Tiers:
    - Flagship Tier 1: 10 stores
    - Regional Mall Tier 2: 25 stores
    - Strip/Outlet Tier 3: 15 stores

    Regions:
    - NorthEast, Central, SouthEast, West (balanced distribution)
    
    Anomaly Injection:
    - 4 Discount-Addicted stores (chronic markdown dependency)
    - 3 Shrinkage Anomaly stores (elevated theft/leakage)
    - 5 High-Turn / Capital-Efficient Flagships (top GMROI, lean inventory)
    - 38 Normal operational stores
    """
    store_ids = [f"STORE_{i:03d}" for i in range(101, 151)]
    
    # Tier allocations: 10 Tier 1, 25 Tier 2, 15 Tier 3
    tiers = (
        ["Flagship Tier 1"] * 10
        + ["Regional Mall Tier 2"] * 25
        + ["Strip/Outlet Tier 3"] * 15
    )
    
    # Regions cycle evenly across the 50 stores
    regions = ["NorthEast", "Central", "SouthEast", "West"]
    region_assignments = [regions[i % 4] for i in range(50)]
    
    # Anomaly profiles
    # Select 5 High-Turn Flagships from Tier 1 stores (indices 0..9)
    high_turn_flagships = {"STORE_101", "STORE_103", "STORE_105", "STORE_107", "STORE_109"}
    
    # Select 3 Shrinkage Anomaly stores (urban / high leakage)
    shrinkage_anomalies = {"STORE_114", "STORE_126", "STORE_138"}
    
    # Select 4 Discount-Addicted stores across Tiers 2 & 3
    discount_addicted = {"STORE_118", "STORE_129", "STORE_135", "STORE_147"}
    
    anomaly_profiles = []
    base_sales_ranges = []
    
    for s_id, tier in zip(store_ids, tiers):
        if s_id in high_turn_flagships:
            profile = "High-Turn Flagship"
            base_sales = rng.uniform(550_000, 720_000)
        elif s_id in shrinkage_anomalies:
            profile = "Shrinkage Anomaly"
            base_sales = rng.uniform(260_000, 380_000)
        elif s_id in discount_addicted:
            profile = "Discount-Addicted"
            base_sales = rng.uniform(220_000, 320_000)
        else:
            profile = "Normal"
            if tier == "Flagship Tier 1":
                base_sales = rng.uniform(480_000, 650_000)
            elif tier == "Regional Mall Tier 2":
                base_sales = rng.uniform(240_000, 360_000)
            else:  # Strip/Outlet Tier 3
                base_sales = rng.uniform(110_000, 180_000)
                
        anomaly_profiles.append(profile)
        base_sales_ranges.append(base_sales)
        
    store_df = pd.DataFrame({
        "store_id": store_ids,
        "region": region_assignments,
        "store_tier": tiers,
        "anomaly_profile": anomaly_profiles,
        "base_monthly_sales": base_sales_ranges,
    })
    
    return store_df


def get_seasonal_multiplier(month_num: int, rng: np.random.Generator) -> float:
    """
    Calculate retail sales seasonality multipliers:
    - Q4 Holiday Peaks (November/December: +35% to +50% lift)
    - Post-Holiday Lull (January/February: -20% dip)
    - Spring Refresh (March/April: +15% to +20% lift)
    - Summer / Back-to-School (May-October: baseline 0.95 - 1.05)
    """
    if month_num == 11:  # November
        return float(rng.uniform(1.35, 1.48))
    elif month_num == 12:  # December
        return float(rng.uniform(1.42, 1.55))
    elif month_num in (1, 2):  # January, February
        return float(rng.uniform(0.78, 0.82))
    elif month_num in (3, 4):  # March, April
        return float(rng.uniform(1.15, 1.20))
    elif month_num in (7, 8):  # July, August (Back to school / summer clearance)
        return float(rng.uniform(1.02, 1.08))
    else:  # May, June, September, October
        return float(rng.uniform(0.96, 1.04))


def generate_rim_dataset(
    seed: int = 42,
    start_date: str = "2024-11-01",
    periods: int = 24,
) -> pd.DataFrame:
    """
    Generates the complete 50-store x 24-month RIM dataset with strict roll-forward balance integrity.
    """
    rng = set_reproducible_seed(seed)
    store_metadata = define_store_network(rng)
    
    # 24 monthly periods
    dates = pd.date_range(start_date, periods=periods, freq="MS")
    
    all_records: List[Dict] = []
    
    for _, store in store_metadata.iterrows():
        store_id = store["store_id"]
        region = store["region"]
        tier = store["store_tier"]
        profile = store["anomaly_profile"]
        base_sales = store["base_monthly_sales"]
        
        # Initial Stock-to-Sales setup for Month 0 (November 2024 - Holiday Peak entry)
        # Retail apparel typically holds 2.5 to 3.5 months of forward stock
        # Flagships with high turn hold leaner stock (~2.0 - 2.4 months)
        if profile == "High-Turn Flagship":
            stock_sales_ratio = rng.uniform(2.0, 2.3)
            init_cost_ratio = rng.uniform(0.39, 0.42)
        elif profile == "Discount-Addicted":
            stock_sales_ratio = rng.uniform(3.0, 3.6)
            init_cost_ratio = rng.uniform(0.42, 0.45)
        else:
            stock_sales_ratio = rng.uniform(2.5, 3.2)
            init_cost_ratio = rng.uniform(0.40, 0.44)
            
        current_beg_inv_retail = base_sales * 1.4 * stock_sales_ratio  # scaled for Q4 inventory build
        current_beg_inv_cost = current_beg_inv_retail * init_cost_ratio
        
        for p_idx, p_date in enumerate(dates):
            month_num = p_date.month
            seasonal_mult = get_seasonal_multiplier(month_num, rng)
            
            # 1. Net Sales POS Revenue
            # Base sales modulated by seasonality and stochastic variance
            store_noise = rng.normal(1.0, 0.03)
            if profile == "High-Turn Flagship":
                store_noise += 0.05  # strong customer velocity
            net_sales = float(np.round(base_sales * seasonal_mult * store_noise, 2))
            
            # 2. Promotional Markdowns ($)
            # Benchmark: 12% - 20% of net sales, higher in clearance months (Jan/Jul)
            # Anomaly: Discount-Addicted stores have chronic markdowns > 35%
            # High-Turn Flagships have lean markdowns: 6% - 10%
            is_clearance_month = month_num in (1, 2, 7)
            if profile == "Discount-Addicted":
                # Chronic markdown dependency (>35% of sales)
                md_rate = rng.uniform(0.36, 0.48)
            elif profile == "High-Turn Flagship":
                md_rate = rng.uniform(0.06, 0.10) if not is_clearance_month else rng.uniform(0.09, 0.12)
            else:
                md_rate = rng.uniform(0.18, 0.24) if is_clearance_month else rng.uniform(0.12, 0.18)
                
            promo_markdowns = float(np.round(net_sales * md_rate, 2))
            
            # 3. Shrinkage ($)
            # Benchmark: 1.2% to 2.8% of retail sales
            # Anomaly: Shrinkage Anomaly stores experience 4.5% to 6.5% volatile shrinkage
            if profile == "Shrinkage Anomaly":
                shrink_rate = rng.uniform(0.045, 0.065)
            elif profile == "High-Turn Flagship":
                shrink_rate = rng.uniform(0.010, 0.016)  # tight store security / loss prevention
            else:
                shrink_rate = rng.uniform(0.012, 0.028)
                
            shrinkage = float(np.round(net_sales * shrink_rate, 2))
            
            # 4. Inbound Purchases ($)
            # Replenishment orders planned to maintain target inventory forward cover
            # Forward look for next month's seasonality
            next_month = (month_num % 12) + 1
            next_seasonal = get_seasonal_multiplier(next_month, rng)
            target_ending_retail = (base_sales * next_seasonal) * stock_sales_ratio * rng.uniform(0.95, 1.05)
            
            # Total goods available retail required = Reductions + Target Ending
            reductions_est = net_sales + promo_markdowns + shrinkage
            implied_purchases_retail = max(target_ending_retail + reductions_est - current_beg_inv_retail, net_sales * 0.6)
            purchases_retail = float(np.round(implied_purchases_retail, 2))
            
            # Initial Markup (IMU %): Target between 55% and 62%
            # Cost-to-Retail on Inbound Purchases = 1 - IMU = 38% to 45%
            # For Discount-Addicted stores, off-price sourcing and vendor concessions yield slightly lower IMU (53%-56%)
            if profile == "Discount-Addicted":
                imu_pct = rng.uniform(0.53, 0.57)
            elif profile == "High-Turn Flagship":
                imu_pct = rng.uniform(0.60, 0.64)  # premium assortment IMU
            else:
                imu_pct = rng.uniform(0.55, 0.62)
                
            inbound_cost_ratio = 1.0 - imu_pct
            purchases_cost = float(np.round(purchases_retail * inbound_cost_ratio, 2))
            
            # 5. Retail Inventory Method (RIM) Balance Sheet Roll-Forward
            # Cumulative Total Goods Available for Sale (TGAS)
            tgas_retail = current_beg_inv_retail + purchases_retail
            tgas_cost = current_beg_inv_cost + purchases_cost
            
            # Cost-to-Retail Ratio (Cost Complement / Cumulative Markon Complement)
            # C/R = Total Cost Available / Total Retail Available
            cost_to_retail_ratio = float(tgas_cost / tgas_retail)
            
            # Total Retail Reductions
            total_reductions = net_sales + promo_markdowns + shrinkage
            
            # Ending Inventory at Retail
            ending_inv_retail = float(np.round(tgas_retail - total_reductions, 2))
            
            # Ending Inventory at Cost (Standard RIM valuation)
            ending_inv_cost = float(np.round(ending_inv_retail * cost_to_retail_ratio, 2))
            
            # 6. Derived Performance & Forensics Metrics
            # COGS under RIM accounting:
            # COGS = TGAS Cost - Ending Inv Cost
            cogs = float(np.round(tgas_cost - ending_inv_cost, 2))
            gross_margin = float(np.round(net_sales - cogs, 2))
            gross_margin_pct = float(gross_margin / net_sales) if net_sales > 0 else 0.0
            
            # Average Inventory at Cost for period
            avg_inv_cost = (current_beg_inv_cost + ending_inv_cost) / 2.0
            gmroi_period = float(gross_margin / avg_inv_cost) if avg_inv_cost > 0 else 0.0
            
            record = {
                "period_date": p_date.strftime("%Y-%m-%d"),
                "store_id": store_id,
                "region": region,
                "store_tier": tier,
                "anomaly_profile": profile,
                # RIM Core Balance Sheet & Flow Variables
                "beginning_inv_cost": float(np.round(current_beg_inv_cost, 2)),
                "beginning_inv_retail": float(np.round(current_beg_inv_retail, 2)),
                "purchases_cost": purchases_cost,
                "purchases_retail": purchases_retail,
                "net_sales": net_sales,
                "promo_markdowns": promo_markdowns,
                "shrinkage": shrinkage,
                "ending_inv_retail": ending_inv_retail,
                "cost_to_retail_ratio": float(np.round(cost_to_retail_ratio, 6)),
                "ending_inv_cost": ending_inv_cost,
                # Key Valuation & Forensics Indicators
                "cogs": cogs,
                "gross_margin": gross_margin,
                "gross_margin_pct": float(np.round(gross_margin_pct, 4)),
                "markdown_pct_sales": float(np.round(promo_markdowns / net_sales, 4)),
                "shrink_pct_sales": float(np.round(shrinkage / net_sales, 4)),
                "imu_pct": float(np.round(imu_pct, 4)),
                "gmroi_monthly": float(np.round(gmroi_period, 4)),
            }
            all_records.append(record)
            
            # Balance Roll-Forward: Month T Ending == Month T+1 Beginning
            current_beg_inv_retail = ending_inv_retail
            current_beg_inv_cost = ending_inv_cost
            
    df = pd.DataFrame(all_records)
    return df


def verify_dataset_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Automated accounting verification checks to guarantee enterprise-grade data integrity.
    """
    errors: List[str] = []
    
    # 1. Row count check: exactly 50 stores * 24 months = 1,200 rows
    if len(df) != 1200:
        errors.append(f"Row count mismatch: expected 1,200, got {len(df)}")
        
    # 2. Balance roll-forward check per store
    for store_id, group in df.groupby("store_id"):
        group_sorted = group.sort_values("period_date").reset_index(drop=True)
        for i in range(len(group_sorted) - 1):
            curr_end_retail = group_sorted.loc[i, "ending_inv_retail"]
            next_beg_retail = group_sorted.loc[i + 1, "beginning_inv_retail"]
            curr_end_cost = group_sorted.loc[i, "ending_inv_cost"]
            next_beg_cost = group_sorted.loc[i + 1, "beginning_inv_cost"]
            
            if abs(curr_end_retail - next_beg_retail) > 0.01:
                errors.append(
                    f"Temporal break at {store_id} period {group_sorted.loc[i, 'period_date']}: "
                    f"Ending Retail {curr_end_retail} != Next Beg Retail {next_beg_retail}"
                )
            if abs(curr_end_cost - next_beg_cost) > 0.01:
                errors.append(
                    f"Temporal break at {store_id} period {group_sorted.loc[i, 'period_date']}: "
                    f"Ending Cost {curr_end_cost} != Next Beg Cost {next_beg_cost}"
                )
                
    # 3. Non-negative inventory check
    if (df["ending_inv_retail"] <= 0).any():
        errors.append("Encountered negative or zero ending inventory retail.")
    if (df["ending_inv_cost"] <= 0).any():
        errors.append("Encountered negative or zero ending inventory cost.")
        
    # 4. RIM mathematical formula validation
    computed_ratio = (df["beginning_inv_cost"] + df["purchases_cost"]) / (
        df["beginning_inv_retail"] + df["purchases_retail"]
    )
    ratio_diff = (df["cost_to_retail_ratio"] - computed_ratio).abs()
    if (ratio_diff > 1e-4).any():
        errors.append(f"Cost-to-Retail ratio formula discrepancy (max diff: {ratio_diff.max()})")
        
    return len(errors) == 0, errors


def print_executive_sanity_summary(df: pd.DataFrame) -> None:
    """Print an executive summary of dataset distributions and anomaly verification."""
    print("=" * 80)
    print("           CAPSTONE PHASE 3: RETAIL COMMAND CENTER DATASET AUDIT")
    print("=" * 80)
    
    total_sales = df["net_sales"].sum()
    total_cogs = df["cogs"].sum()
    total_gross_margin = df["gross_margin"].sum()
    overall_gm_pct = total_gross_margin / total_sales
    
    avg_cr_ratio = df["cost_to_retail_ratio"].mean()
    avg_imu = df["imu_pct"].mean()
    
    # Network GMROI: Total Gross Margin / Network Average Inventory Cost
    # Calculate average inventory per store across all months, then sum
    avg_inv_cost_per_store = df.groupby("store_id")["ending_inv_cost"].mean().sum()
    annualized_gmroi = (total_gross_margin / 2.0) / avg_inv_cost_per_store  # 2-year horizon annualized
    
    print(f"Dataset Shape               : {df.shape[0]:,} rows x {df.shape[1]} columns (50 stores x 24 months)")
    print(f"Date Range                  : {df['period_date'].min()} to {df['period_date'].max()}")
    print(f"Cumulative Network Sales    : ${total_sales:,.2f}")
    print(f"Cumulative Gross Margin     : ${total_gross_margin:,.2f} ({overall_gm_pct:.1%})")
    print(f"Average Cost-to-Retail (C/R): {avg_cr_ratio:.4f} (Cost Complement)")
    print(f"Average Inbound IMU Target  : {avg_imu:.1%}")
    print(f"Network Annualized GMROI    : {annualized_gmroi:.2f}x")
    print("-" * 80)
    
    print("PERFORMANCE PROFILE BREAKDOWN (CLUSTER ANOMALY FORENSICS):")
    profile_agg = df.groupby("anomaly_profile").agg(
        store_count=("store_id", "nunique"),
        avg_monthly_sales=("net_sales", "mean"),
        avg_markdown_pct=("markdown_pct_sales", "mean"),
        avg_shrink_pct=("shrink_pct_sales", "mean"),
        avg_gm_pct=("gross_margin_pct", "mean"),
        avg_cost_ratio=("cost_to_retail_ratio", "mean"),
    )
    print(profile_agg.to_string(formatters={
        "avg_monthly_sales": "${:,.0f}".format,
        "avg_markdown_pct": "{:.1%}".format,
        "avg_shrink_pct": "{:.1%}".format,
        "avg_gm_pct": "{:.1%}".format,
        "avg_cost_ratio": "{:.4f}".format,
    }))
    print("-" * 80)
    
    print("SAMPLE ANOMALY STORE PROFILES:")
    sample_stores = ["STORE_101", "STORE_118", "STORE_126", "STORE_110"]
    sample_df = df[df["store_id"].isin(sample_stores)].groupby(["store_id", "anomaly_profile", "store_tier"]).agg(
        total_sales=("net_sales", "sum"),
        total_markdowns=("promo_markdowns", "sum"),
        markdown_rate=("markdown_pct_sales", "mean"),
        shrink_rate=("shrink_pct_sales", "mean"),
        gross_margin_pct=("gross_margin_pct", "mean"),
    ).reset_index()
    print(sample_df.to_string(formatters={
        "total_sales": "${:,.0f}".format,
        "total_markdowns": "${:,.0f}".format,
        "markdown_rate": "{:.1%}".format,
        "shrink_rate": "{:.1%}".format,
        "gross_margin_pct": "{:.1%}".format,
    }))
    print("=" * 80)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    csv_path = output_dir / "synthetic_apparel_rim_data.csv"
    parquet_path = output_dir / "synthetic_apparel_rim_data.parquet"
    
    print("Generating synthetic apparel RIM dataset (50 stores x 24 months)...")
    df = generate_rim_dataset(seed=42, start_date="2024-11-01", periods=24)
    
    # Verify balance sheet roll-forward integrity
    print("Running enterprise balance sheet and accounting verification...")
    is_valid, errors = verify_dataset_integrity(df)
    if not is_valid:
        print("ERROR: Accounting integrity verification failed:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    print("[OK] All balance sheet roll-forward and mathematical integrity checks PASSED.")
    
    # Export artifacts
    df.to_csv(csv_path, index=False)
    print(f"[OK] Exported CSV: {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB)")
    
    df.to_parquet(parquet_path, index=False)
    print(f"[OK] Exported Parquet: {parquet_path.name} ({parquet_path.stat().st_size / 1024:.1f} KB)")
    
    # Executive console summary
    print_executive_sanity_summary(df)


if __name__ == "__main__":
    main()
