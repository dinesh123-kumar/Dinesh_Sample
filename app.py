from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



# Initialize Flask app
app = Flask(__name__)

# Load trained models
with open('lg_model_5.pkl', 'rb') as file:
    logreg_model = pickle.load(file)

with open('rf_model_5.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('svm_model_5.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('model_5.pkl', 'rb') as file:
    deep_model = pickle.load(file)

# Initialize MinMaxScaler with fixed feature range based on training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit([
    [350, 18, 0, 0, 1, 0],    # Example min values from training
    [850, 92, 10, 250000, 4, 200000]  # Example max values from training
])

# Define the correct feature columns based on training
feature_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    data = request.form

    # Extract and convert form values
    CreditScore = float(data['CreditScore'])
    Age = float(data['Age'])
    Tenure = float(data['Tenure'])
    Balance = float(data['Balance'])
    NumOfProducts = float(data['NumOfProducts'])
    EstimatedSalary = float(data['EstimatedSalary'])
    HasCrCard = int(data['HasCrCard'])  # Binary: 1 or 0
    IsActiveMember = int(data['IsActiveMember'])  # Binary: 1 or 0

    # Convert Gender
    Gender = 1 if data['Gender'].lower() == 'male' else 0

    # Handle Geography one-hot encoding dynamically
    Geography = data['Geography']
    geography_one_hot = {
        'Geography_France': 1 if Geography == 'France' else 0,
        'Geography_Germany': 1 if Geography == 'Germany' else 0,
        'Geography_Spain': 1 if Geography == 'Spain' else 0
    }

    # Prepare input in correct order for models
    input_data = pd.DataFrame([[
        CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
        HasCrCard, IsActiveMember, EstimatedSalary,
        geography_one_hot['Geography_France'],
        geography_one_hot['Geography_Germany'],
        geography_one_hot['Geography_Spain']
    ]], columns=feature_columns)

    # Apply scaling on numerical features
    input_data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.transform(
        input_data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']]
    )

    # Generate Predictions
    logreg_pred = logreg_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]
    deep_pred = deep_model.predict(input_data)[0][0] > 0.5  # Keras model binary output

    # Prepare response
    response = {
        'Logistic Regression': int(logreg_pred),
        'Random Forest': int(rf_pred),
        'SVM': int(svm_pred),
        'Deep Learning': int(deep_pred)
    }

    return render_template('index.html', prediction_text=response)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Render provides 'PORT'
    app.run(host='0.0.0.0', port=port)
