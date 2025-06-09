import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Define model architecture matching the training script
class MushroomNet(nn.Module):
    def __init__(self, input_dim):
        super(MushroomNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 2)  # Two output units

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output(x)  # Raw logits

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Determine input dimension from encoders
input_dim = len(label_encoders)

# Load the trained model
model = MushroomNet(input_dim)
model.load_state_dict(torch.load("mushroom_model.pth"))
model.eval()

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]

        # Convert input features to encoded format
        encoded = []
        for i, (col, le) in enumerate(label_encoders.items()):
            val = features.get(col)
            if val is None:
                return jsonify({"error": f"Missing value for: {col}"}), 400
            if val not in le.classes_:
                return jsonify({"error": f"Invalid value '{val}' for feature '{col}'"}), 400
            encoded_val = le.transform([val])[0]
            encoded.append(encoded_val)

        input_tensor = torch.tensor([encoded], dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            label = "Poisonous" if predicted_class == 1 else "Edible"
            confidence = probs[0, predicted_class].item()

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
