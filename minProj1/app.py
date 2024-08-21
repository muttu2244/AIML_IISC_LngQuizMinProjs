from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('minProj1/iris_model.pkl')

@app.route('/')
def home():
    return "Welcome to the ML Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if data and 'features' in data:
        prediction = model.predict([data['features']])
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
