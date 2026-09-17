"""
Retail Inventory Method (RIM) Calculation Engine & Financial Forensics Module.

Executes deterministic retail inventory accounting mathematics, multi-period
balance sheet roll-forwards, and forensic audit diagnostics for enterprise
merchandise financial planning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def compute_period_rim(
    beg_cost: float,
    beg_retail: float,
    purchases_cost: float,
    purchases_retail: float,
    net_sales: float,
    markdowns: float,
    shrinkage: float,
    net_markups: float = 0.0,
) -> Dict[str, float]:
    """
    Calculates deterministic single-period RIM accounting values.

    Args:
        beg_cost: Starting inventory asset value at cost.
        beg_retail: Starting inventory value at original ticket retail.
        purchases_cost: Inbound vendor/DC deliveries at cost.
        purchases_retail: Inbound vendor/DC deliveries marked up to ticket retail.
        net_sales: Realized point-of-sale top-line revenue.
        markdowns: Promotional and clearance markdowns taken during period.
        shrinkage: Book-to-physical inventory shrinkage (theft, damage, error).
        net_markups: Additional upward price revisions to original retail.

    Returns:
        Dictionary containing all calculated RIM balance sheet and flow figures.
    """
    # 1. Total Goods Available for Sale (Cost & Retail)
    goods_avail_cost = float(beg_cost + purchases_cost)
    goods_avail_retail = float(beg_retail + purchases_retail + net_markups)

    # 2. Cost-to-Retail Ratio (Cost Complement / Cumulative Markon %)
    # Graceful zero-division handling
    if goods_avail_retail > 0.0:
        cost_to_retail_ratio = float(goods_avail_cost / goods_avail_retail)
    else:
        cost_to_retail_ratio = 0.0

    # 3. Ending Inventory at Retail (Book Value)
    total_reductions = float(net_sales + markdowns + shrinkage)
    ending_inv_retail = float(goods_avail_retail - total_reductions)

    # 4. Ending Inventory at Cost (Balance Sheet Valuation)
    # Enforces zero-floor guard on cost valuation
    ending_inv_cost = max(0.0, float(ending_inv_retail * cost_to_retail_ratio))

    # 5. Cost of Goods Sold (COGS) & Margin Forensics
    gross_cogs = float(goods_avail_cost - ending_inv_cost)
    gross_margin = float(net_sales - gross_cogs)
    gross_margin_pct = (
        float((gross_margin / net_sales) * 100.0) if net_sales > 0.0 else 0.0
    )

    # Average inventory at cost for period
    avg_inv_cost = float((beg_cost + ending_inv_cost) / 2.0)
    gmroi = float(gross_margin / avg_inv_cost) if avg_inv_cost > 0.0 else 0.0

    # Operational flow ratios
    markdown_pct = float((markdowns / net_sales) * 100.0) if net_sales > 0.0 else 0.0
    shrink_pct = float((shrinkage / net_sales) * 100.0) if net_sales > 0.0 else 0.0

    return {
        "goods_avail_cost": goods_avail_cost,
        "goods_avail_retail": goods_avail_retail,
        "cost_to_retail_ratio": cost_to_retail_ratio,
        "total_reductions": total_reductions,
        "ending_inv_retail": ending_inv_retail,
        "ending_inv_cost": ending_inv_cost,
        "gross_cogs": gross_cogs,
        "gross_margin": gross_margin,
        "gross_margin_pct": gross_margin_pct,
        "avg_inv_cost": avg_inv_cost,
        "gmroi": gmroi,
        "markdown_pct_sales": markdown_pct,
        "shrink_pct_sales": shrink_pct,
    }


class RIMEngine:
    """
    Enterprise-grade Retail Inventory Method pipeline for auditing,
    multi-period balance sheet roll-forward, and forensic diagnostics.
    """

    def __init__(self, tolerance: float = 0.02) -> None:
        """
        Initialize the RIM engine with an accounting dollar-penny tolerance threshold.
        """
        self.tolerance = tolerance

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized / row-wise re-calculation of RIM metrics across a full DataFrame.
        Adds recalculated columns with prefix 'calc_'.
        """
        result = df.copy()

        # Extract core inputs with safe defaults
        beg_cost = result["beginning_inv_cost"].astype(float)
        beg_retail = result["beginning_inv_retail"].astype(float)
        purch_cost = result["purchases_cost"].astype(float)
        purch_retail = result["purchases_retail"].astype(float)
        net_sales = result["net_sales"].astype(float)
        markdowns = result["promo_markdowns"].astype(float)
        shrinkage = result["shrinkage"].astype(float)
        net_markups = (
            result["net_markups"].astype(float)
            if "net_markups" in result.columns
            else 0.0
        )

        # 1. Total Goods Available
        calc_goods_cost = beg_cost + purch_cost
        calc_goods_retail = beg_retail + purch_retail + net_markups

        # 2. Cost-to-Retail Ratio (Zero-division safe)
        calc_cr_ratio = np.where(
            calc_goods_retail > 0.0, calc_goods_cost / calc_goods_retail, 0.0
        )

        # 3. Reductions & Ending Retail
        calc_reductions = net_sales + markdowns + shrinkage
        calc_ending_retail = calc_goods_retail - calc_reductions

        # 4. Ending Cost with zero-floor
        calc_ending_cost = np.maximum(0.0, calc_ending_retail * calc_cr_ratio)

        # 5. COGS, Margins, and Forensics
        calc_cogs = calc_goods_cost - calc_ending_cost
        calc_gross_margin = net_sales - calc_cogs
        calc_gross_margin_pct = np.where(
            net_sales > 0.0, (calc_gross_margin / net_sales) * 100.0, 0.0
        )
        calc_avg_inv_cost = (beg_cost + calc_ending_cost) / 2.0
        calc_gmroi = np.where(
            calc_avg_inv_cost > 0.0, calc_gross_margin / calc_avg_inv_cost, 0.0
        )
        calc_md_pct = np.where(
            net_sales > 0.0, (markdowns / net_sales) * 100.0, 0.0
        )
        calc_shrink_pct = np.where(
            net_sales > 0.0, (shrinkage / net_sales) * 100.0, 0.0
        )

        # Assign calculated columns
        result["calc_goods_avail_cost"] = calc_goods_cost.round(2)
        result["calc_goods_avail_retail"] = calc_goods_retail.round(2)
        result["calc_cost_to_retail_ratio"] = calc_cr_ratio.round(6)
        result["calc_total_reductions"] = calc_reductions.round(2)
        result["calc_ending_inv_retail"] = calc_ending_retail.round(2)
        result["calc_ending_inv_cost"] = calc_ending_cost.round(2)
        result["calc_cogs"] = calc_cogs.round(2)
        result["calc_gross_margin"] = calc_gross_margin.round(2)
        result["calc_gross_margin_pct"] = calc_gross_margin_pct.round(2)
        result["calc_gmroi"] = calc_gmroi.round(4)
        result["calc_markdown_pct"] = calc_md_pct.round(2)
        result["calc_shrink_pct"] = calc_shrink_pct.round(2)

        return result

    def audit_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Audits an existing retail dataset against exact RIM mathematical formulas
        and checks temporal balance roll-forward integrity.

        Returns:
            Audited DataFrame with diagnostic discrepancy columns and an audit summary dict.
        """
        audited_df = self.evaluate_dataframe(df)

        # Compare reported vs calculated values if reported values exist
        comparisons = [
            ("ending_inv_retail", "calc_ending_inv_retail", "delta_ending_retail"),
            ("ending_inv_cost", "calc_ending_inv_cost", "delta_ending_cost"),
            ("cost_to_retail_ratio", "calc_cost_to_retail_ratio", "delta_cr_ratio"),
            ("cogs", "calc_cogs", "delta_cogs"),
            ("gross_margin", "calc_gross_margin", "delta_gross_margin"),
        ]

        discrepancies: Dict[str, int] = {}
        for reported_col, calc_col, delta_col in comparisons:
            if reported_col in audited_df.columns:
                delta = (audited_df[reported_col] - audited_df[calc_col]).abs()
                audited_df[delta_col] = delta.round(4)
                threshold = 1e-4 if "ratio" in delta_col else self.tolerance
                discrepancies[delta_col] = int((delta > threshold).sum())

        # Check multi-period balance roll-forward breaks per store
        balance_breaks: List[Dict[str, Any]] = []
        if "store_id" in audited_df.columns and "period_date" in audited_df.columns:
            for store_id, store_group in audited_df.groupby("store_id"):
                store_sorted = store_group.sort_values("period_date").reset_index(
                    drop=True
                )
                for i in range(len(store_sorted) - 1):
                    end_retail = float(store_sorted.loc[i, "ending_inv_retail"])
                    next_beg_retail = float(
                        store_sorted.loc[i + 1, "beginning_inv_retail"]
                    )
                    end_cost = float(store_sorted.loc[i, "ending_inv_cost"])
                    next_beg_cost = float(
                        store_sorted.loc[i + 1, "beginning_inv_cost"]
                    )

                    retail_diff = abs(end_retail - next_beg_retail)
                    cost_diff = abs(end_cost - next_beg_cost)

                    if (
                        retail_diff > self.tolerance
                        or cost_diff > self.tolerance
                    ):
                        balance_breaks.append(
                            {
                                "store_id": store_id,
                                "period_t": store_sorted.loc[i, "period_date"],
                                "period_t_plus_1": store_sorted.loc[
                                    i + 1, "period_date"
                                ],
                                "retail_diff": round(retail_diff, 2),
                                "cost_diff": round(cost_diff, 2),
                            }
                        )

        audit_summary = {
            "total_rows_audited": len(audited_df),
            "discrepancies": discrepancies,
            "temporal_rollforward_breaks_count": len(balance_breaks),
            "temporal_rollforward_breaks": balance_breaks,
            "passed": (
                all(count == 0 for count in discrepancies.values())
                and len(balance_breaks) == 0
            ),
        }

        return audited_df, audit_summary

    def rollforward_store(
        self,
        store_activity_df: pd.DataFrame,
        initial_cost: float,
        initial_retail: float,
    ) -> pd.DataFrame:
        """
        Executes a dynamic multi-period balance sheet roll-forward for a single store,
        propagating Month T's ending balance to Month T+1's beginning balance.

        Args:
            store_activity_df: DataFrame with monthly periods containing:
                'purchases_cost', 'purchases_retail', 'net_sales', 'promo_markdowns', 'shrinkage'
            initial_cost: Starting inventory cost for Period 0.
            initial_retail: Starting inventory retail for Period 0.

        Returns:
            Complete DataFrame with calculated roll-forward balances.
        """
        sorted_activity = store_activity_df.copy()
        if "period_date" in sorted_activity.columns:
            sorted_activity = sorted_activity.sort_values(
                "period_date"
            ).reset_index(drop=True)

        current_beg_cost = float(initial_cost)
        current_beg_retail = float(initial_retail)

        rollforward_rows: List[Dict[str, Any]] = []

        for _, row in sorted_activity.iterrows():
            row_dict = row.to_dict()
            purch_cost = float(row_dict.get("purchases_cost", 0.0))
            purch_retail = float(row_dict.get("purchases_retail", 0.0))
            sales = float(row_dict.get("net_sales", 0.0))
            mds = float(row_dict.get("promo_markdowns", 0.0))
            shrink = float(row_dict.get("shrinkage", 0.0))
            net_markups = float(row_dict.get("net_markups", 0.0))

            period_res = compute_period_rim(
                beg_cost=current_beg_cost,
                beg_retail=current_beg_retail,
                purchases_cost=purch_cost,
                purchases_retail=purch_retail,
                net_sales=sales,
                markdowns=mds,
                shrinkage=shrink,
                net_markups=net_markups,
            )

            record = {
                **row_dict,
                "beginning_inv_cost": round(current_beg_cost, 2),
                "beginning_inv_retail": round(current_beg_retail, 2),
                **{k: (round(v, 2) if "ratio" not in k and "pct" not in k and "gmroi" not in k else v) for k, v in period_res.items()},
            }
            rollforward_rows.append(record)

            # Roll forward balances: Month T ending becomes Month T+1 beginning
            current_beg_cost = period_res["ending_inv_cost"]
            current_beg_retail = period_res["ending_inv_retail"]

        return pd.DataFrame(rollforward_rows)

    def diagnose_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Screens store performance for operational anomalies and merchandise forensics:
        - Discount-Addicted: Chronic promotional markdown dependency (>35% of sales)
        - Shrinkage Anomaly: Elevated inventory leakage/theft (>4.0% of sales)
        - Capital-Efficient Flagship: High GMROI (>network median), lean markdowns (<12%), GM% > 55%
        - Normal: Baseline operational profile

        Returns:
            Store-level aggregated diagnostic summary with assigned forensic tags.
        """
        # Ensure calculated fields exist
        eval_df = self.evaluate_dataframe(df)

        store_summary = eval_df.groupby(["store_id", "region", "store_tier"]).agg(
            total_net_sales=("net_sales", "sum"),
            total_cogs=("calc_cogs", "sum"),
            total_gross_margin=("calc_gross_margin", "sum"),
            total_markdowns=("promo_markdowns", "sum"),
            total_shrinkage=("shrinkage", "sum"),
            avg_ending_cost=("calc_ending_inv_cost", "mean"),
            avg_cost_ratio=("calc_cost_to_retail_ratio", "mean"),
        ).reset_index()

        store_summary["gross_margin_pct"] = (
            store_summary["total_gross_margin"] / store_summary["total_net_sales"]
        ) * 100.0
        store_summary["markdown_pct_sales"] = (
            store_summary["total_markdowns"] / store_summary["total_net_sales"]
        ) * 100.0
        store_summary["shrink_pct_sales"] = (
            store_summary["total_shrinkage"] / store_summary["total_net_sales"]
        ) * 100.0

        # Annualized GMROI assuming 2-year horizon (24 months)
        store_summary["annualized_gmroi"] = (
            (store_summary["total_gross_margin"] / 2.0)
            / store_summary["avg_ending_cost"]
        ).round(2)

        median_gmroi = store_summary["annualized_gmroi"].median()

        # Classification rule
        def classify_store(row: pd.Series) -> str:
            if row["markdown_pct_sales"] >= 35.0:
                return "Discount-Addicted"
            elif row["shrink_pct_sales"] >= 4.0:
                return "Shrinkage Anomaly"
            elif (
                row["markdown_pct_sales"] < 12.0
                and row["gross_margin_pct"] >= 55.0
                and row["annualized_gmroi"] >= median_gmroi
            ):
                return "High-Turn Flagship"
            else:
                return "Normal"

        store_summary["detected_profile"] = store_summary.apply(
            classify_store, axis=1
        )
        return store_summary


def run_engine_audit(parquet_path: Union[str, Path]) -> None:
    """Executes a forensic audit on the provided dataset and prints the executive report."""
    path = Path(parquet_path)
    if not path.exists():
        print(f"Error: Dataset file not found at {path}")
        sys.exit(1)

    print(f"Loading retail dataset from {path.name}...")
    df = pd.read_parquet(path)

    engine = RIMEngine(tolerance=0.02)
    audited_df, summary = engine.audit_dataframe(df)

    print("=" * 80)
    print("         RETAIL INVENTORY METHOD (RIM) ENGINE: AUDIT & FORENSICS")
    print("=" * 80)
    print(f"Dataset Size            : {summary['total_rows_audited']:,} records")
    print(f"Discrepancies Detected  : {summary['discrepancies']}")
    print(f"Balance Roll-Forward Brk: {summary['temporal_rollforward_breaks_count']}")
    print(f"Overall Audit Status    : {'[PASS]' if summary['passed'] else '[FAIL]'}")
    print("-" * 80)

    # Anomaly Diagnostics
    print("STORE-LEVEL ANOMALY DIAGNOSTIC SUMMARY:")
    diagnostics = engine.diagnose_anomalies(df)
    cluster_counts = diagnostics["detected_profile"].value_counts().to_dict()
    for profile, count in cluster_counts.items():
        print(f"  * {profile:<22}: {count} stores")

    print("-" * 80)
    print("DETAILED PERFORMANCE BY DETECTED PROFILE:")
    profile_view = diagnostics.groupby("detected_profile").agg(
        store_count=("store_id", "count"),
        avg_total_sales=("total_net_sales", "mean"),
        avg_gm_pct=("gross_margin_pct", "mean"),
        avg_markdown_pct=("markdown_pct_sales", "mean"),
        avg_shrink_pct=("shrink_pct_sales", "mean"),
        avg_annual_gmroi=("annualized_gmroi", "mean"),
        avg_cost_ratio=("avg_cost_ratio", "mean"),
    )
    print(
        profile_view.to_string(
            formatters={
                "avg_total_sales": "${:,.0f}".format,
                "avg_gm_pct": "{:.1f}%".format,
                "avg_markdown_pct": "{:.1f}%".format,
                "avg_shrink_pct": "{:.1f}%".format,
                "avg_annual_gmroi": "{:.2f}x".format,
                "avg_cost_ratio": "{:.4f}".format,
            }
        )
    )
    print("=" * 80)


if __name__ == "__main__":
    default_parquet = Path(__file__).resolve().parent / "synthetic_apparel_rim_data.parquet"
    run_engine_audit(default_parquet)
