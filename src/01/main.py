import sys
import pandas as pd
import numpy as np
from visualize import show_fit

#-------- Setup --------
# python -m venv .venv
# source .venv/bin/activate  # MacOS/Linux
# .\.venv\Scripts\activate   # Windows
# pip install -r requirements.txt

#-------- CSVS --------
# Example CSVs:
# - salaries.csv (Hard)
# - housing.csv (Medium)
# - studying.csv (Easy)

#-------- Usage --------
# python main.py [csv_name] [COL_X] [COL_Y]

#-------- What you need to fix --------
# Fill in the blanks (____) and remove comments (--- IGNORE ---)
# Fill in the blanks at lines 33, 60, 73.

# -------- CLI / Defaults --------
CSV_PATH = "data/raw/"
CSV_PATH += sys.argv[1] if len(sys.argv) > 1 else "salaries.csv"
COL_X    = sys.argv[2] if len(sys.argv) > 2 else "Years of Experience"
COL_Y    = sys.argv[3] if len(sys.argv) > 3 else "Salary"

# -------- Load & peek --------
# Create the dataframe by loading the CSV (what function can you use)
# df = pd.____(CSV_PATH)

print("\n=== HEAD ===")
print(df.head().to_string(index=False))

print("\n=== INFO ===")
df.info()

print("\n=== DESCRIBE ===")
print(df.describe(include="all").transpose().to_string())

# -------- Pick two columns --------
if COL_X not in df.columns or COL_Y not in df.columns:
    print(f"\nSet valid columns. Available: {list(df.columns)}")
    sys.exit(1)

data = df[[COL_X, COL_Y]].copy()

# -------- Clean & drop NAs --------
for col in [COL_X, COL_Y]:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.replace(",", "").str.replace("$", "").str.replace("%", ""),
        errors="coerce"
    )

before = len(data)

# ~~~~~~~~~ drop rows in data with NAs (what function can you use) ~~~~~~~~~

print(f"\nDropped {before - len(data)} rows after cleaning & removing NAs.")

# -------- Regression --------
x = data[COL_X].to_numpy()
y = data[COL_Y].to_numpy()

if len(x) < 2:
    raise ValueError("Not enough data points to fit a line.")

# ~~~~~~~~~ fit a line to x, y (what function can you use) ~~~~~~~~~
# ~~~~~~~~~ store slope in m and intercept in b ~~~~~~~~~
# Hint: m, b = np.____(____, ____, 1)

print(f"\nRegression: {COL_Y} ≈ {m:.4f} * {COL_X} + {b:.4f}")

# -------- Plot --------
show_fit(
    x=x,
    y=y,
    m=m,
    b=b,
    title=f"{COL_X} vs {COL_Y} (Linear Fit)",
    xlabel=COL_X,
    ylabel=COL_Y,
)