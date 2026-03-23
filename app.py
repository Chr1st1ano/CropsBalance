from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained crop yield model
model = joblib.load('crop_yield_model.pkl')

# Load the label encoders
country_encoder = joblib.load('Country_encoder.pkl')
region_encoder = joblib.load('Region_encoder.pkl')
crop_type_encoder = joblib.load('Crop_Type_encoder.pkl')
adaptation_encoder = joblib.load('Adaptation_Strategies_encoder.pkl')

# Load feature order
feature_order = joblib.load('feature_order.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Validate inputs
            required_fields = ['year', 'country', 'region', 'crop_type', 'avg_temp', 'precipitation', 'co2_emissions', 'extreme_weather', 'irrigation_access', 'pesticide_use', 'fertilizer_use', 'soil_health', 'adaptation', 'economic_impact']
            for field in required_fields:
                if field not in request.form or request.form[field].strip() == '':
                    return render_template('result.html', prediction=f"Validation Error: Missing required field '{field}'", error=True)

            # Get inputs from form
            year = int(request.form['year'])
            country = request.form['country']
            region = request.form['region']
            crop_type = request.form['crop_type']
            avg_temp = float(request.form['avg_temp'])
            precipitation = float(request.form['precipitation'])
            co2_emissions = float(request.form['co2_emissions'])
            extreme_weather = int(request.form['extreme_weather'])
            irrigation_access = float(request.form['irrigation_access'])
            pesticide_use = float(request.form['pesticide_use'])
            fertilizer_use = float(request.form['fertilizer_use'])
            soil_health = float(request.form['soil_health'])
            adaptation = request.form['adaptation']
            economic_impact = float(request.form['economic_impact'])

            # Encode categorical variables
            country_encoded = country_encoder.transform([country])[0]
            region_encoded = region_encoder.transform([region])[0]
            crop_type_encoded = crop_type_encoder.transform([crop_type])[0]
            adaptation_encoded = adaptation_encoder.transform([adaptation])[0]

            # Create input array in correct order
            input_data = np.array([[
                year, country_encoded, region_encoded, crop_type_encoded, avg_temp,
                precipitation, co2_emissions, extreme_weather, irrigation_access,
                pesticide_use, fertilizer_use, soil_health, adaptation_encoded, economic_impact
            ]])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Convert prediction to readable format
            yield_levels = {'Low': 'Low Yield', 'Medium': 'Medium Yield', 'High': 'High Yield'}
            result = yield_levels.get(prediction, prediction)

            return render_template('result.html',
                                 prediction=result,
                                 year=year,
                                 country=country,
                                 crop_type=crop_type)

        except Exception as e:
            return render_template('result.html',
                                 prediction=f"Error: {str(e)}",
                                 error=True)

if __name__ == '__main__':
    app.run(debug=True)