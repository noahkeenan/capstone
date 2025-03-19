import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('athleticism_predictor_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None, status=None, draft_round=None, name=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data for the prediction
        input_data = [
            float(request.form['40_yard_dash']),
            float(request.form['bench_press']),
            float(request.form['vertical'])
        ]

        # Reshape the input data to be 2D (1 sample, 3 features)
        input_data = np.array(input_data).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_data)[0]

        # Render the result on the page
        return render_template('index.html', prediction=prediction, status=None, draft_round=None, name=None)

    except Exception as e:
        return f"Error: {e}"

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the athletic score from the form
        athletic_score = float(request.form['athletic_score'])

        # Classify based on the new athletic score ranges
        if athletic_score <= 55:
            status = "Bust"
            draft_round = "Don't draft"
        elif 56 <= athletic_score <= 75:
            status = "Average"
            draft_round = "Rounds 5-7"
        elif 76 <= athletic_score <= 83:
            status = "Good"
            draft_round = "Rounds 2-4"
        else:
            status = "Boom"
            draft_round = "Round 1"

        # Render the result on the page
        return render_template('index.html', prediction=None, status=status, draft_round=draft_round, name=request.form['name'])

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
