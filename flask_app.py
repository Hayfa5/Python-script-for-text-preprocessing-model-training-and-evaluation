from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import finalpreprocess
from models import train_logistic_regression, train_neural_network, tfidf_vectorizer, lr_model  # Assurez-vous d'importer lr_model
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No "text" key in request data'})

    input_text = data['text']

    # Preprocess input text
    cleaned_text = finalpreprocess(input_text)
    
    # Convert input text to vector using TF-IDF vectorizer
    input_vector = tfidf_vectorizer.transform([cleaned_text])

    # Make prediction using trained model
    predicted_category = lr_model.predict(input_vector)[0]

    # Convert predicted category to string
    predicted_category_str = str(predicted_category)

    return jsonify({'predicted_category': predicted_category_str})

if __name__ == '__main__':
    app.run(debug=True)
