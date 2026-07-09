import pandas as pd
from explore_constituents import get_sp500_membership

membership = get_sp500_membership()

# Stocks that LEFT the index (have an end date)
left = membership[membership["end"].notna()].copy()
print(f"Stocks that left the index: {len(left)}")

# What does IsDelisted look like for them?
print("\nIsDelisted value counts among stocks that left:")
print(left["is_delisted"].value_counts(dropna=False))

# Show a sample of each type so we can eyeball whether the flag
# actually distinguishes failures from acquisitions
print("\n--- Sample: is_delisted = 1 ---")
sample1 = left[left["is_delisted"] == 1].head(15)
for _, row in sample1.iterrows():
    print(f"  {row['ticker']:8} {str(row['name'])[:35]:35} left {str(row['end'])[:10]}")

print("\n--- Sample: is_delisted = 0 ---")
sample0 = left[left["is_delisted"] == 0].head(15)
for _, row in sample0.iterrows():
    print(f"  {row['ticker']:8} {str(row['name'])[:35]:35} left {str(row['end'])[:10]}")