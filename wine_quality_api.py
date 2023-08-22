# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:22:00 2023

@author: m8abb
"""

from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Load the trained models
red_model = load('red_wine_quality_model.joblib')
white_model = load('white_wine_quality_model.joblib')

@app.route('/')
def index():
    return "Hello, this is my Flask app!"

@app.route('/predict/red', methods=['POST'])
def predict_red():
    data = request.get_json(force=True)
    prediction = red_model.predict([data['features']])
    return jsonify(prediction=int(prediction[0]))

@app.route('/predict/white', methods=['POST'])
def predict_white():
    data = request.get_json(force=True)
    prediction = white_model.predict([data['features']])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
