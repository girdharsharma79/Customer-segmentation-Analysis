import pandas as pd
import numpy as np

np.random.seed(7)

n = 200  # matches the classic Kaggle Mall Customer Segmentation dataset size

# Build 5 natural customer "clusters" (like the real mall dataset) so
# K-Means finds meaningful, textbook segments
segments = [
    # (n_customers, income_range, spending_range, age_range)
    (40, (15, 40), (70, 100), (18, 32)),   # young, low income, high spending
    (40, (15, 40), (0, 30),  (30, 60)),    # low income, low spending
    (40, (40, 75), (40, 60), (25, 55)),    # mid income, mid spending (average)
    (40, (75, 140),(70, 100),(20, 45)),    # high income, high spending
    (40, (75, 140),(0, 35),  (35, 70)),    # high income, low spending (careful)
]

rows = []
cid = 1
for count, income_range, spend_range, age_range in segments:
    for _ in range(count):
        age = np.random.randint(*age_range)
        income = round(np.random.uniform(*income_range), 1)
        spending = int(np.clip(np.random.uniform(*spend_range), 1, 99))
        gender = np.random.choice(["Male", "Female"])
        rows.append({
            "CustomerID": cid,
            "Gender": gender,
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spending
        })
        cid += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
df["CustomerID"] = range(1, len(df) + 1)

# inject light real-world messiness for the cleaning step
missing_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_idx, "Annual Income (k$)"] = np.nan
dupe_rows = df.sample(6, random_state=3)
df = pd.concat([df, dupe_rows], ignore_index=True)

df.to_csv("raw_customer_data.csv", index=False)
print(df.shape)
print(df.head())
