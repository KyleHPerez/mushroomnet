const { useState } = React;

const featureMap = {
  cap_shape: {
    label: "cap-shape",
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
    label: "cap-surface",
    options: {
      f: "Fibrous",
      g: "Grooves",
      y: "Scaly",
      s: "Smooth",
    },
  },
  cap_color: {
    label: "cap-color",
    options: {
      // n: "Brown",
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
  bruises: {
    label: "bruises",
    options: {
      t: "Bruises",
      f: "None",
    },
  },
  odor: {
    label: "odor",
    options: {
      a: "Almond",
      l: "Anise",
      c: "Creosote",
      y: "Fishy",
      m: "Musty",
      n: "None",
      p: "Pungent",
      s: "Spicy",
    },
  },
  gill_attachment: {
    label: "gill-attachment",
    options: {
      a: "Attached",
      d: "Descending",
      f: "Free",
      n: "Notched",
    },
  },
  gill_spacing: {
    label: "gill-spacing",
    options: {
      c: "Close",
      w: "Crowded",
      d: "Distant",
    },
  },
  gill_size: {
    label: "gill-size",
    options: {
      b: "Broad",
      n: "Narrow",
    },
  },
  gill_color: {
    label: "gill-color",
    options: {
      k: "Black",
      // n: "Brown",
      b: "Buff",
      h: "Chocolate",
      g: "Gray",
      r: "Green",
      o: "Orange",
      p: "Pink",
      u: "Purple",
      e: "Red",
      w: "White",
      y: "Yellow",
    },
  },
  stalk_shape: {
    label: "stalk-shape",
    options: {
      e: "Enlarging",
      t: "Tapering",
    },
  },
  stalk_root: {
    label: "stalk-root",
    options: {
      b: "Bulbous",
      c: "Club",
      u: "Cup",
      e: "Equal",
      z: "Rhizomorphs",
      r: "Rooted",
    },
  },
  stalk_surface_above_ring: {
    label: "stalk-surface-above-ring",
    options: {
      f: "Fibrous",
      y: "Scaly",
      k: "Silky",
      s: "Smooth",
    },
  },
  stalk_surface_below_ring: {
    label: "stalk-surface-below-ring",
    options: {
      f: "Fibrous",
      y: "Scaly",
      k: "Silky",
      s: "Smooth",
    },
  },
  stalk_color_above_ring: {
    label: "stalk-color-above-ring",
    options: {
      // n: "Brown",
      b: "Buff",
      g: "Gray",
      o: "Orange",
      p: "Pink",
      e: "Red",
      w: "White",
      y: "Yellow",
    },
  },
  stalk_color_below_ring: {
    label: "stalk-color-below-ring",
    options: {
      // n: "Brown",
      b: "Buff",
      g: "Gray",
      o: "Orange",
      p: "Pink",
      e: "Red",
      w: "White",
      y: "Yellow",
    },
  },
  veil_type: {
    label: "veil-type",
    options: {
      p: "Partial",
      u: "Universal",
    },
  },
  veil_color: {
    label: "veil-color",
    options: {
      // n: "Brown",
      o: "Orange",
      w: "White",
      y: "Yellow",
    },
  },
  ring_number: {
    label: "ring-number",
    options: {
      n: "None",
      o: "One",
      t: "Two",
    },
  },
  ring_type: {
    label: "ring-type",
    options: {
      // c: "Cobwebby",
      e: "Evanescent",
      // f: "Flaring",
      l: "Large",
      n: "None",
      p: "Pendant",
      s: "Sheathing",
      z: "Zone",
    },
  },
  spore_print_color: {
    label: "spore-print-color",
    options: {
      k: "Black",
      // n: "Brown",
      b: "Buff",
      h: "Chocolate",
      r: "Green",
      o: "Orange",
      u: "Purple",
      w: "White",
      y: "Yellow",
    },
  },
  population: {
    label: "population",
    options: {
      a: "Abundant",
      c: "Clustered",
      n: "Numerous",
      s: "Scattered",
      v: "Several",
      y: "Solitary",
    },
  },
  habitat: {
    label: "habitat",
    options: {
      g: "Grasses",
      l: "Leaves",
      p: "Paths",
      u: "Urban",
      w: "Waste",
      d: "Woods",
    },
  },
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

  const snakeToHyphen = (str) => str.replace(/_/g, "-");
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isFormComplete) {
      alert("Please fill out all fields.");
      return;
    }

    const formattedData = {};
    for (const key in formData) {
      const hyphenKey = snakeToHyphen(key);  // e.g., cap_shape -> cap-shape
      formattedData[hyphenKey] = formData[key];
    }

    console.log({ features: formattedData });

    try {
      const response = await fetch("https://mushroomnet.fly.dev/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: formattedData })
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

// render manually into the root div
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<MushroomForm />);
