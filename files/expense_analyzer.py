"""
Expense Pattern Analyzer
------------------------
Reads an expense CSV (date, category, amount, description) and finds:
  1. Spending by category
  2. Monthly trend + simple next-month forecast
  3. Weekday spending pattern
  4. Unusual (anomalous) expenses using z-score
  5. Budget alerts and saving tips

Usage:
    python expense_analyzer.py                 # uses sample data
    python expense_analyzer.py my_expenses.csv # uses your own file

CSV columns required: date (YYYY-MM-DD), category, amount
Optional column: description
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "output"
SAMPLE_FILE = "sample_expenses.csv"

# Monthly budget limits per category (edit as needed)
BUDGETS = {
    "Food": 6000,
    "Transport": 2500,
    "Shopping": 4000,
    "Bills": 5000,
    "Entertainment": 2000,
    "Health": 1500,
}


# ------------------------------------------------------------ Data
def generate_sample_data(path=SAMPLE_FILE, seed=42):
    """Create six months of realistic sample expenses."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180)
    profile = {  # category: (daily probability, mean, std)
        "Food": (0.9, 220, 70),
        "Transport": (0.6, 110, 40),
        "Shopping": (0.15, 900, 400),
        "Bills": (0.05, 1500, 500),
        "Entertainment": (0.2, 450, 200),
        "Health": (0.05, 600, 250),
    }
    rows = []
    for d in dates:
        for cat, (p, mean, std) in profile.items():
            if rng.random() < p:
                amt = max(20, rng.normal(mean, std))
                # weekends: a bit more food and entertainment
                if d.dayofweek >= 5 and cat in ("Food", "Entertainment"):
                    amt *= 1.4
                rows.append((d.date(), cat, round(amt, 2), f"{cat} expense"))
    # a few unusual expenses
    for i in rng.choice(len(rows), 4, replace=False):
        d, c, a, desc = rows[i]
        rows[i] = (d, c, round(a * 5, 2), f"{c} (large)")
    pd.DataFrame(rows, columns=["date", "category", "amount", "description"]) \
        .to_csv(path, index=False)
    print(f"Sample data created: {path}")


def load_data(path):
    df = pd.read_csv(path)
    required = {"date", "category", "amount"}
    if not required.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {', '.join(sorted(required))}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "category"])
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["month"] = df["date"].dt.to_period("M")
    df["weekday"] = df["date"].dt.day_name()
    return df


# ------------------------------------------------------------ Analysis
def category_summary(df):
    s = df.groupby("category")["amount"].agg(["sum", "mean", "count"])
    s["share_%"] = (s["sum"] / s["sum"].sum() * 100).round(1)
    return s.sort_values("sum", ascending=False).round(2)


def complete_months(df):
    """Months fully covered by the data (partial first/last months distort trends)."""
    months = sorted(df["month"].unique())
    if df["date"].min().day > 1 and len(months) > 1:
        months = months[1:]
    if df["date"].max() < df["date"].max().to_period("M").end_time.normalize() \
            and len(months) > 1:
        months = months[:-1]
    return months


def monthly_trend(df):
    full = df[df["month"].isin(complete_months(df))]
    return full.groupby("month")["amount"].sum()


def forecast_next_month(monthly):
    """Linear-regression forecast on monthly totals."""
    if len(monthly) < 3:
        return None
    y = monthly.values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return max(0, slope * len(y) + intercept), slope


def weekday_pattern(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    return df.groupby("weekday")["amount"].mean().reindex(order).round(2)


def find_anomalies(df, threshold=2.5):
    """Flag expenses far above the usual amount for their category."""
    g = df.groupby("category")["amount"]
    df = df.copy()
    df["z"] = (df["amount"] - g.transform("mean")) / g.transform("std")
    return df[df["z"] > threshold].sort_values("z", ascending=False)


def budget_alerts(df):
    last = complete_months(df)[-1]
    spent = df[df["month"] == last].groupby("category")["amount"].sum()
    alerts = []
    for cat, limit in BUDGETS.items():
        if cat in spent and spent[cat] > limit:
            alerts.append((cat, spent[cat], limit))
    return last, alerts


# ------------------------------------------------------------ Charts
def make_charts(df, monthly, weekday, cat_sum):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    cat_sum["sum"].plot.pie(ax=ax[0, 0], autopct="%1.0f%%", ylabel="")
    ax[0, 0].set_title("Spending by Category")

    monthly.index = monthly.index.astype(str)
    monthly.plot(ax=ax[0, 1], marker="o", color="tab:blue")
    ax[0, 1].set_title("Monthly Spending Trend")
    ax[0, 1].set_ylabel("Amount")

    weekday.plot.bar(ax=ax[1, 0], color="tab:orange")
    ax[1, 0].set_title("Average Spend by Weekday")
    ax[1, 0].set_xlabel("")
    ax[1, 0].tick_params(axis="x", rotation=45)

    pivot = df.pivot_table(index="month", columns="category",
                           values="amount", aggfunc="sum").fillna(0)
    pivot.index = pivot.index.astype(str)
    pivot.plot.bar(stacked=True, ax=ax[1, 1])
    ax[1, 1].set_title("Monthly Spending by Category")
    ax[1, 1].set_xlabel("")
    ax[1, 1].tick_params(axis="x", rotation=45)
    ax[1, 1].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "expense_report.png")
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# ------------------------------------------------------------ Report
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_FILE
    if not os.path.exists(path):
        if path == SAMPLE_FILE:
            generate_sample_data(path)
        else:
            sys.exit(f"File not found: {path}")

    df = load_data(path)
    lines = []
    out = lines.append

    out("=" * 55)
    out("        EXPENSE PATTERN ANALYZER REPORT")
    out("=" * 55)
    out(f"Period : {df['date'].min().date()}  to  {df['date'].max().date()}")
    out(f"Total  : {df['amount'].sum():,.2f}   ({len(df)} transactions)")

    cat_sum = category_summary(df)
    out("\n[1] SPENDING BY CATEGORY")
    out(cat_sum.to_string())
    top = cat_sum.index[0]
    out(f"-> Highest spending: {top} ({cat_sum.loc[top, 'share_%']}% of total)")

    monthly = monthly_trend(df)
    out("\n[2] MONTHLY TREND (complete months only)")
    out(monthly.round(2).to_string())
    result = forecast_next_month(monthly)
    if result:
        pred, slope = result
        direction = "increasing" if slope > 0 else "decreasing"
        out(f"-> Spending is {direction} by about {abs(slope):,.0f} per month")
        out(f"-> Forecast for next month: {pred:,.2f}")

    weekday = weekday_pattern(df)
    out("\n[3] WEEKDAY PATTERN (average per transaction)")
    out(weekday.to_string())
    out(f"-> You spend most on: {weekday.idxmax()}")

    anomalies = find_anomalies(df)
    out("\n[4] UNUSUAL EXPENSES")
    if anomalies.empty:
        out("No unusual expenses found.")
    else:
        for _, r in anomalies.head(8).iterrows():
            out(f"  {r['date'].date()}  {r['category']:<14}{r['amount']:>10,.2f}"
                f"   (z-score {r['z']:.1f})")

    last, alerts = budget_alerts(df)
    out(f"\n[5] BUDGET CHECK for {last}")
    if alerts:
        for cat, spent, limit in alerts:
            out(f"  ! {cat}: spent {spent:,.0f} vs budget {limit:,.0f} "
                f"(over by {spent - limit:,.0f})")
        out("-> Tip: reduce spending in the categories above.")
    else:
        out("  All categories are within budget. Great job!")

    chart = make_charts(df, monthly, weekday, cat_sum)
    out(f"\nChart saved: {chart}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
