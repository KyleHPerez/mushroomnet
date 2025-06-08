import pandas as pd

# Define column names from UCI docs
columns = [
    "target", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Load raw CSV
df = pd.read_csv("mushrooms.csv", header=None, names=columns)

# Drop rows with missing values (indicated by "?")
df = df[~df.eq('?').any(axis=1)]

# Save cleaned file
df.to_csv("mushrooms_cleaned.csv", index=False)
print(f"Cleaned dataset saved with {len(df)} rows.")
