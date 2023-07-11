import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('checkpoints/model.pkl')

# Load the Iris dataset
dataset = pd.read_csv('Iris.csv')

# Extract the features from the dataset
features = dataset.drop('species', axis=1)

# Preprocess the features
features = np.array(features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template("index.html", prediction_text='Predicted Iris species: {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000, debug=True)

