# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model1.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form
    
    City = str(data['City'])
    type1 = str(data['type'])
    condition = str(data['condition'])
    city_area = str(data['city_area'])
    Area = int(data['Area'])
    hasElevator = int(data.get('hasElevator', 0))
    hasParking = int(data.get('hasParking', 0))
    hasBars = int(data.get('hasBars', 0))
    hasStorage = int(data.get('hasStorage', 0))
    hasBalcony = int(data.get('hasBalcony', 0))
    hasMamad = int(data.get('hasMamad', 0))
    handicapFriendly = int(data.get('handicapFriendly', 0))
    floor = int(data['floor'])

    # Create a feature array

    data = {'Area': Area,'hasElevator ': hasElevator, 'hasParking ': hasParking,
            'hasBars ': hasBars,'hasStorage ': hasStorage,'hasBalcony ': hasBalcony,
            'hasMamad ': hasMamad,'handicapFriendly ': handicapFriendly, 
            'floor': floor,'City': City, 'type': type1,
            'city_area': city_area, 'condition ': condition}
    df = pd.DataFrame(data, index = [0])
    
    # Make a prediction
    y_pred = round(model.predict(df)[0])

    # Return the predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run(debug=True)
