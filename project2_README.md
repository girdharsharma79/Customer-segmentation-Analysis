# Customer Segmentation Analysis

Segments mall customers into distinct groups based on income and spending
behavior using K-Means clustering — so a business knows exactly which
customer types to target with marketing.

## What it does

1. **Load** — reads a customer dataset (Customer ID, Gender, Age, Annual
   Income, Spending Score). A synthetic generator is included so the
   project runs standalone; swap in the real
   [Kaggle Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
   if you'd rather use real data.
2. **Clean & preprocess** — removes duplicate customers, fills missing
   income with the median, standardizes gender labels.
3. **Explore** — prints the distribution of income, spending score, and
   gender split.
4. **Cluster** — scales the features and runs K-Means, scanning k=2..8
   with the elbow method (`elbow_plot.png`) before fitting the final
   5-cluster model.
5. **Visualize & interpret** — plots the clusters on an income-vs-spending
   scatter plot (`customer_clusters.png`) and labels each one in plain
   English (e.g. "High Income - High Spending (Target)").

## Files

| File | Description |
|---|---|
| `customer_segmentation.py` | Full pipeline: load → clean → explore → cluster → visualize |
| `gen_customer_data.py` | Generates the synthetic sample dataset |
| `requirements.txt` | Python dependencies |
| `raw_customer_data.csv` | Raw sample dataset (created on run) |
| `clean_customer_data.csv` | Cleaned dataset (created on run) |
| `customer_segments.csv` | Final dataset with cluster ID + segment label (created on run) |
| `elbow_plot.png` | Elbow-method chart used to justify k=5 (created on run) |
| `customer_clusters.png` | Static scatter plot of the final clusters (created on run) |
| `customer_dashboard.html` | Interactive dashboard — filter by segment/gender |

## Run it

```bash
pip install -r requirements.txt
python gen_customer_data.py        # generates raw_customer_data.csv
python customer_segmentation.py    # runs the full pipeline
```

Then open `customer_dashboard.html` in a browser for the interactive view.

## Sample results

On the generated dataset (200 customers):

| Segment | Customers | Avg. income | Avg. spending |
|---|---|---|---|
| High Income - High Spending (Target) | 38 | $108.5k | 86.3 |
| High Income - Low Spending (Careful) | 39 | $110.4k | 17.7 |
| Low Income - High Spending (Impulsive) | 42 | $29.9k | 86.2 |
| Low Income - Low Spending (Budget) | 40 | $28.2k | 13.4 |
| Average Income - Average Spending | 41 | $57.2k | 49.2 |

**Business takeaway:** the "High Income - High Spending" and "Low Income -
High Spending" groups are the best marketing targets (they already spend
freely); "High Income - Low Spending" customers are a growth opportunity
if the business can win their trust.

## Tech stack

- **pandas** — data loading, cleaning
- **scikit-learn** — `StandardScaler` + `KMeans` clustering
- **matplotlib** — elbow plot and cluster scatter plot
- **Chart.js** (in the HTML dashboard) — interactive scatter plot

## Possible extensions

- Swap in the real Kaggle Mall Customer dataset
- Try clustering on more features (e.g. Age) with 3D visualization
- Compare K-Means against hierarchical clustering or DBSCAN
