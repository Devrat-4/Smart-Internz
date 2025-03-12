import numpy as np
import pickle
import pandas as pd
import os
import logging
from flask import Flask, request, render_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the trained Random Forest model
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file '{MODEL_PATH}' not found. Ensure it exists.")
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it exists.")

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
logging.info("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract user input
        input_features = [float(x) for x in request.form.values()]
        features_values = np.array([input_features])

        # Define feature names
        feature_names = ['holiday', 'temp', 'rain', 'snow', 'weather',
                         'day', 'month', 'year', 'hours', 'minutes', 'seconds']

        # Convert to DataFrame
        data = pd.DataFrame(features_values, columns=feature_names)

        # Make prediction
        prediction = model.predict(data)[0]
        prediction_text = f"Estimated Traffic Volume: {prediction:.2f}"
        logging.info("Prediction made successfully.")

    except ValueError:
        logging.error("Invalid input: Please enter numeric values.")
        prediction_text = "Invalid input: Please enter numeric values."
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)