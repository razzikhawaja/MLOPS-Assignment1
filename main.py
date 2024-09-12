from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


app = Flask(__name__)

model = joblib.load('house_price_model.pkl')


@app.route('/')
def home():
 
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    
    med_inc = float(request.form['MedInc'])
    house_age = float(request.form['HouseAge'])
    ave_rooms = float(request.form['AveRooms'])
    ave_bedrms = float(request.form['AveBedrms'])
    population = float(request.form['Population'])
    ave_occup = float(request.form['AveOccup'])
    latitude = float(request.form['Latitude'])
    longitude = float(request.form['Longitude'])
    
 
    input_data = pd.DataFrame([{
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }])
    
    
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
