# test_pipeline.py
from src.agents import run_pipeline

with open("data/sample_pdfs/GPAY.pdf", "rb") as f:
    result = run_pipeline(f, monthly_income=15000)

if result["error"]:
    print(f"Error: {result['error']}")
else:
    import pandas as pd
    df = pd.DataFrame(result["categorized_df"])
    print("=== TRANSACTIONS ===")
    print(df[["date","description","amount","category","method"]].to_string())

    r = result["risk_report"]
    print(f"\n=== RISK REPORT ===")
    print(f"Total spent:  ₹{r['total_spent']}")
    print(f"Risk level:   {r['risk_level']} ({r['risk_score']}%)")
    print(f"Savings left: ₹{r['savings_potential']}")

    print(f"\n=== AI SUGGESTIONS ===")
    print(result["suggestions"])