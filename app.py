from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def new():
    return render_template('new.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                    'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
                    'Functioning Day', 'Holiday', 'Seasons']

        data = {}
        for feature in features:
            data[feature] = float(request.form.get(feature, 0))

        rf_model = load('rf_model.joblib')
        xgb_model = load('xgb_model.joblib')
        lgbm_model = load('lgbm_model.joblib')

        data_for_prediction = [data[feature] for feature in features]

        rf_prediction = rf_model.predict([data_for_prediction])
        xgb_prediction = xgb_model.predict([data_for_prediction])
        lgbm_prediction = lgbm_model.predict([data_for_prediction])

        weight_rf = 0.5
        weight_xgb = 0.2
        weight_nn = 0.3

        ensemble_prediction = (weight_rf * rf_prediction) + (weight_xgb * xgb_prediction) + (weight_nn * lgbm_prediction)

        functioning_day = 'Functioning Day' in request.form
        print("Functioning Day:", functioning_day)

        if not functioning_day:
            ensemble_prediction = '0'

        return render_template('new.html', ensemble_prediction=ensemble_prediction)

    return render_template('new.html')

if __name__ == "__main__":
    app.run(debug=True, port=8080)
