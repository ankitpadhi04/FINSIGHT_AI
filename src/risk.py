# src/risk.py

import pandas as pd

RISK_THRESHOLDS = {
    "low":    0.50,
    "medium": 0.75,
}

# What % of income each category should ideally not exceed
CATEGORY_LIMITS = {
    "Food":          0.15,
    "Travel":        0.10,
    "Shopping":      0.10,
    "Entertainment": 0.05,
    "Health":        0.10,
    "Bills":         0.20,
    "Groceries":     0.15,
    "Education":     0.10,
    "Transfer":      0.20,
    "Investment":    0.20,
}


def calculate_risk(df: pd.DataFrame, monthly_income: float) -> dict:
    """
    Takes categorized transactions DataFrame and monthly income.
    Returns full risk report dict.
    """
    if df.empty or monthly_income <= 0:
        return _empty_report()

    # Only analyse debits
    debits = df[df["transaction_type"] == "debit"].copy()
    total_spent = debits["amount"].sum()
    spend_ratio = total_spent / monthly_income

    # Per category breakdown
    category_summary = (
        debits.groupby("category")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total"})
    )
    category_summary["percentage"] = (
        category_summary["total"] / monthly_income * 100
    ).round(1)
    category_summary["ratio"] = category_summary["total"] / monthly_income

    # Overall risk level
    if spend_ratio <= RISK_THRESHOLDS["low"]:
        risk_level = "Low"
        risk_color = "green"
        risk_score = round(spend_ratio * 100, 1)
    elif spend_ratio <= RISK_THRESHOLDS["medium"]:
        risk_level = "Medium"
        risk_color = "orange"
        risk_score = round(spend_ratio * 100, 1)
    else:
        risk_level = "High"
        risk_color = "red"
        risk_score = round(spend_ratio * 100, 1)

    # Flag overspent categories
    overspent = []
    for _, row in category_summary.iterrows():
        cat   = row["category"]
        ratio = row["ratio"]
        limit = CATEGORY_LIMITS.get(cat, 0.15)
        if ratio > limit:
            overspent.append({
                "category":  cat,
                "spent":     round(row["total"], 2),
                "limit_pct": round(limit * 100, 1),
                "actual_pct": round(ratio * 100, 1),
                "excess":    round((ratio - limit) * monthly_income, 2)
            })

    # Savings potential
    total_saveable = sum(o["excess"] for o in overspent)
    savings_potential = round(monthly_income - total_spent, 2)

    return {
        "total_spent":       round(total_spent, 2),
        "monthly_income":    monthly_income,
        "spend_ratio":       round(spend_ratio, 4),
        "risk_level":        risk_level,
        "risk_color":        risk_color,
        "risk_score":        risk_score,
        "savings_potential": savings_potential,
        "total_saveable":    round(total_saveable, 2),
        "category_summary":  category_summary.to_dict(orient="records"),
        "overspent":         overspent,
        "transaction_count": len(debits)
    }


def _empty_report() -> dict:
    return {
        "total_spent":       0,
        "monthly_income":    0,
        "spend_ratio":       0,
        "risk_level":        "Unknown",
        "risk_color":        "gray",
        "risk_score":        0,
        "savings_potential": 0,
        "total_saveable":    0,
        "category_summary":  [],
        "overspent":         [],
        "transaction_count": 0
    }