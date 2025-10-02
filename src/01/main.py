import sys
import pandas as pd
import numpy as np
from visualize import show_fit

CSV_PATH = "data/raw/___"  #<<< CHANGE THE BLANK TO YOUR CSV PATH in data/raw >>>

#<<< CHANGE THESE TO YOUR COLUMN NAMES >>>
COL_X = "___"   # e.g., "hours_studied"
COL_Y = "___"   # e.g., "exam_score"

#1) Load CSV
df = pd.read_csv(CSV_PATH)

#If columns not set, show options and exit
if COL_X == "___" or COL_Y == "___" or COL_X not in df.columns or COL_Y not in df.columns:
    print("\nSet COL_X and COL_Y to valid column names from your CSV.")
    print("Available columns:", list(df.columns))
    sys.exit(1)

#2) Quick look at the dataset
print("\n=== HEAD ===")
print(df.head().to_string(index=False))

print("\n=== INFO ===")
df.info()

print("\n=== DESCRIBE (numeric) ===")
print(df.describe(numeric_only=True).to_string())

# 3)Keep just the two columns we care about and drop missing rows
data = df[[COL_X, COL_Y]].copy()
data = data.dropna(subset=[COL_X, COL_Y])

# 4)Change em to numeric (anything non-numeric becomes NaN), then drop NAs again
data[COL_X] = pd.to_numeric(data[COL_X], errors="coerce")
data[COL_Y] = pd.to_numeric(data[COL_Y], errors="coerce")
before = len(data)
data = data.dropna(subset=[COL_X, COL_Y])
print(f"\nDropped {before - len(data)} rows after coercing to numeric and removing NAs.")

# 5)Remove simple outliers with z-scores (|z| <= 3)
x_mean, x_std = data[COL_X].mean(), data[COL_X].std(ddof=0)
y_mean, y_std = data[COL_Y].mean(), data[COL_Y].std(ddof=0)

#Avoid divide-by-zero if variance is 0
if x_std == 0 or y_std == 0:
    print("\nWarning: zero variance detected; skipping outlier filtering.")
else:
    z_x = (data[COL_X] - x_mean) / x_std
    z_y = (data[COL_Y] - y_mean) / y_std
    kept = (z_x.abs() <= 3) & (z_y.abs() <= 3)
    removed = (~kept).sum()
    data = data[kept]
    print(f"Removed {removed} potential outliers using |z| <= 3.")

#6)Fit the line of best fit: y ≈ m*x + b
x = data[COL_X].to_numpy()
y = data[COL_Y].to_numpy()
m, b = np.polyfit(x, y, 1)
print(f"\nRegression: {COL_Y} ≈ {m:.4f} * {COL_X} + {b:.4f}")

#7) Plot
show_fit(
    x=x,
    y=y,
    m=m,
    b=b,
    title=f"{COL_X} vs {COL_Y} (Linear Fit)",
    xlabel=COL_X,
    ylabel=COL_Y,
)