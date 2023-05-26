from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
lr_model = joblib.load('lr_model.joblib')

# Load the CountVectorizer
vectorizer = joblib.load('vectorizer.joblib')

# Define API endpoint for classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    complaint = data['complaint']
    complaint_vector = vectorizer.transform([complaint])
    product = lr_model.predict(complaint_vector)[0]
    return jsonify({'product': product})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
