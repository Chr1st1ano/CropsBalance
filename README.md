# Crop Yield Prediction Web Application
Machine Learning Model Deployment using Flask and Jinja2
A Flask web application that predicts crop yield levels (Low, Medium, High) based on agricultural and environmental factors using machine learning.

## Features

- **Home Page**: Comprehensive input form for agricultural parameters
- **Prediction**: Processes multiple environmental and agricultural factors
- **Result Page**: Displays yield prediction with detailed recommendations

## Dataset Features

The model considers the following factors:
- Year, Country, Region, Crop Type
- Temperature, Precipitation, CO2 Emissions
- Extreme Weather Events, Irrigation Access
- Pesticide/Fertilizer Usage, Soil Health
- Adaptation Strategies, Economic Impact

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and go to `http://127.0.0.1:5000/`

3. Fill in the agricultural parameters and click "Predict Crop Yield"

4. View the prediction result with recommendations

## Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~62% on test data
- **Output**: Low (< 1.5 MT/ha), Medium (1.5-3.5 MT/ha), High (> 3.5 MT/ha)

## Files

- `app.py`: Main Flask application with preprocessing
- `crop_yield_model.pk1`: Trained Random Forest model
- `*_encoder.pk1`: Label encoders for categorical features
- `templates/index.html`: Input form for agricultural parameters
- `templates/result.html`: Prediction results with recommendations
- `requirements.txt`: Python dependencies
