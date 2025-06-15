import React, { useState } from "react";

const featureMap = {
  cap_shape: {
    label: "Cap Shape",
    options: {
      b: "Bell",
      c: "Conical",
      x: "Convex",
      f: "Flat",
      k: "Knobbed",
      s: "Sunken",
    },
  },
  cap_surface: {
    label: "Cap Surface",
    options: {
      f: "Fibrous",
      g: "Grooves",
      y: "Scaly",
      s: "Smooth",
    },
  },
  cap_color: {
    label: "Cap Color",
    options: {
      n: "Brown",
      b: "Buff",
      c: "Cinnamon",
      g: "Gray",
      r: "Green",
      p: "Pink",
      u: "Purple",
      e: "Red",
      w: "White",
      y: "Yellow",
    },
  },
  // Add remaining 19 features with the same structure
};

const MushroomForm = () => {
  const initialFormData = Object.keys(featureMap).reduce((acc, key) => {
    acc[key] = null;
    return acc;
  }, {});

  const [formData, setFormData] = useState(initialFormData);
  const [prediction, setPrediction] = useState(null);

  const updateField = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const isFormComplete = Object.values(formData).every((v) => v !== null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isFormComplete) {
      alert("Please fill out all fields.");
      return;
    }

    try {
      const response = await fetch("https://mushroomnet.fly.dev/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error(err);
      setPrediction("Error fetching prediction.");
    }
  };

  return (
    <div className="form-container">
      <h1>Mushroom Classification</h1>
      <form onSubmit={handleSubmit}>
        {Object.entries(featureMap).map(([field, { label, options }]) => (
          <div key={field} className="form-group">
            <label>{label}</label>
            <select
              value={formData[field] || ""}
              onChange={(e) => updateField(field, e.target.value)}
              required
            >
              <option value="" disabled>
                -- select {label.toLowerCase()} --
              </option>
              {Object.entries(options).map(([code, desc]) => (
                <option key={code} value={code}>
                  {desc}
                </option>
              ))}
            </select>
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>
      {prediction && <h2>Prediction: {prediction}</h2>}
    </div>
  );
};

export default MushroomForm;
