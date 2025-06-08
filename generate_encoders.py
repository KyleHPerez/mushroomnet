import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("mushrooms_cleaned.csv")

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Create and fit encoders for features
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save feature encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Create and save target encoder
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("✅ Saved 'label_encoders.pkl' and 'target_encoder.pkl'")
