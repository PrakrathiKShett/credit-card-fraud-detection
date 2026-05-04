from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- Machine Learning Setup ---
# Create synthetic data
data = {
    'Area': [800, 1000, 1200, 1500, 1800, 2000, 850, 1100, 1300, 1600],
    'Location': ['Bangalore', 'Delhi', 'Bangalore', 'Mumbai', 'Mumbai', 'Bangalore', 'Delhi', 'Mumbai', 'Delhi', 'Bangalore'],
    # Base price calculation + some random noise
    # Base: Bangalore=50, Mumbai=60, Delhi=45
    # Per sqft: Bangalore=0.04, Mumbai=0.05, Delhi=0.035
    'Price': [82.0, 80.0, 98.0, 135.0, 150.0, 130.0, 74.75, 115.0, 90.5, 114.0] 
}
df = pd.DataFrame(data)

# Dummy encoding for Location
df_encoded = pd.get_dummies(df, columns=['Location'])

# Features and Target
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save feature columns to ensure input matches during prediction
feature_columns = X.columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        area = float(data.get('area', 0))
        location = data.get('location', '')

        if area <= 0:
            return jsonify({'error': 'Area must be greater than 0'})

        if location not in ['Bangalore', 'Mumbai', 'Delhi']:
            return jsonify({'error': 'Invalid location'})

        # Prepare input data
        input_data = pd.DataFrame(columns=feature_columns)
        input_data.loc[0] = 0  # Initialize with zeros
        
        input_data.at[0, 'Area'] = area
        loc_col = f'Location_{location}'
        if loc_col in input_data.columns:
            input_data.at[0, loc_col] = 1

        # Predict
        predicted_price = model.predict(input_data)[0]
        
        # Add some slight formatting (round to 2 decimal places)
        formatted_price = f"{predicted_price:.2f}"

        return jsonify({
            'success': True,
            'price': formatted_price,
            'location': location,
            'area': area
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
