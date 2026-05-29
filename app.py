from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
print("📂 Loading model and vectorizer...")
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print("✅ Model loaded successfully!")

def clean_text(text):
    """Clean and preprocess text (same function used in training)"""
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_news(news_text):
    """Predict if news is real or fake"""
    # Clean the text
    cleaned_text = clean_text(news_text)
    
    # Convert to numerical features
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Get confidence scores
    if prediction == 1:
        result = "REAL"
        confidence = probability[1] * 100
        fake_confidence = probability[0] * 100
    else:
        result = "FAKE"
        confidence = probability[0] * 100
        fake_confidence = probability[1] * 100
    
    return {
        'result': result,
        'confidence': round(confidence, 2),
        'fake_probability': round(probability[0] * 100, 2),
        'real_probability': round(probability[1] * 100, 2)
    }

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get news text from request
        data = request.get_json()
        news_text = data.get('news_text', '')
        
        if not news_text or len(news_text.strip()) < 20:
            return jsonify({
                'error': 'Please enter at least 20 characters of news text'
            }), 400
        
        # Make prediction
        result = predict_news(news_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)