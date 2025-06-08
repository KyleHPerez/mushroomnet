import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load and encode dataset
df = pd.read_csv("mushrooms_cleaned.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Label encode categorical features
encoders = {}
for column in X.columns:
    enc = LabelEncoder()
    X[column] = enc.fit_transform(X[column])
    encoders[column] = enc

label_enc = LabelEncoder()
y = label_enc.fit_transform(y)  # 'e' -> 0, 'p' -> 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Neural Network class
class MushroomNet(nn.Module):
    def __init__(self):
        super(MushroomNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output(x)

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MushroomNet().to(device)

# Weighted loss to emphasize poisonous class
class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluation with lowered threshold to boost recall
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    probs = torch.softmax(outputs, dim=1)
    threshold = 0.4
    predicted = (probs[:, 1] >= threshold).int().cpu()

print("\nClassification Report (with recall-prioritized thresholding):")
print(classification_report(y_test, predicted, target_names=["Edible", "Poisonous"]))

# Save model for reuse or deployment
torch.save(model.state_dict(), "mushroom_model.pth")
print("Model file saved successfully.")
