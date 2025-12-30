from flask import Flask, render_template, request
import numpy as np
import joblib  # for loading ML model

app = Flask(__name__)

# Load your trained ML model (replace with your model path)
# Example: model = joblib.load("model.joblib")
model = None  # placeholder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        s1_total = int(request.form['s1_total'])
        s1_passed = int(request.form['s1_passed'])
        s1_sgpa = float(request.form['s1_sgpa'])
        s2_total = int(request.form['s2_total'])
        s2_passed = int(request.form['s2_passed'])
        s2_sgpa = float(request.form['s2_sgpa'])
        age = int(request.form['age'])
        tuition = int(request.form['tuition'])

        # Example feature array for model
        features = np.array([[s1_total, s1_passed, s1_sgpa,
                              s2_total, s2_passed, s2_sgpa,
                              age, tuition]])

        # If you have a trained model, uncomment this:
        # prediction = model.predict(features)[0]

        # Temporary dummy prediction
        prediction = "High Risk" if s1_sgpa < 5 or s2_sgpa < 5 else "Low Risk"

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
