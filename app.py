from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
model_path = 'model/fake_review_model.pkl'
tfidf_path = 'model/tfidf_vectorizer.pkl'

if not os.path.exists(model_path):
    print("ERROR: Model file not found!")
    print("   Run this first: python train_model.py")
    sys.exit(1)

if not os.path.exists(tfidf_path):
    print("ERROR: TF-IDF vectorizer not found!")
    print("   Run this first: python train_model.py")
    sys.exit(1)
app = Flask(__name__)
CORS(app)
try:
    print("🔄 Loading model...")
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    print("✅ NLP tools loaded!")
except Exception as e:
    print(f" ERROR loading NLP tools: {e}")
    print("   Run: python -c \"import nltk; nltk.download('stopwords'); nltk.download('wordnet')\"")
    sys.exit(1)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_features(text):
    features = {}

    words = text.split()
    features['char_count'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['excl_per_word'] = features['exclamation_count'] / (features['word_count'] + 1)

    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
    features['caps_word_ratio'] = features['all_caps_words'] / (features['word_count'] + 1)

    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    features['unique_word_ratio'] = len(unique_words) / (len(lower_words) + 1)

    if lower_words:
        word_freq = pd.Series(lower_words).value_counts()
        features['max_word_repeat'] = word_freq.max()
        features['words_repeated_3plus'] = sum(1 for c in word_freq if c >= 3)
    else:
        features['max_word_repeat'] = 0
        features['words_repeated_3plus'] = 0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    features['sentence_count'] = max(len(sentences), 1)
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']

    blob = TextBlob(text)
    features['polarity'] = blob.sentiment.polarity
    features['subjectivity'] = blob.sentiment.subjectivity
    features['extreme_polarity'] = abs(blob.sentiment.polarity)

    fake_words = [
        'amazing', 'perfect', 'best', 'incredible', 'must buy',
        'love', 'wow', 'fantastic', 'miraculous', 'buy now',
        'changed my life', 'ever', 'greatest', 'unbelievable',
        'life changing', 'must have', 'phenomenal', 'flawless',
        'revolutionary', 'miracle', 'superb', 'outstanding',
        'magnificent', 'divine', 'heaven', 'obsessed',
        'game changer', 'hurry', 'limited', 'act now',
        'selling fast', 'last chance', 'urgent', 'guaranteed',
        "don't wait", 'buy buy', 'trust me', 'believe me'
    ]
    text_lower = text.lower()
    features['fake_word_count'] = sum(1 for w in fake_words if w in text_lower)
    features['fake_word_ratio'] = features['fake_word_count'] / (features['word_count'] + 1)

    marketing_words = [
        'innovative', 'premium', 'seamless', 'streamline', 'optimize',
        'cutting edge', 'state of the art', 'world class', 'best in class',
        'unparalleled', 'second to none', 'pinnacle', 'epitome',
        'meticulously', 'craftsmanship', 'transformative', 'revolutionize',
        'indispensable', 'instrumental', 'exceptional', 'superior',
        'unprecedented', 'remarkable', 'extraordinary', 'sophistication',
        'refinement', 'excellence', 'masterfully', 'thoughtful design',
        'stands in a class', 'sets a new standard', 'quantum leap'
    ]
    features['marketing_word_count'] = sum(1 for w in marketing_words if w in text_lower)

    specific_indicators = [
        'inch', 'feet', 'pound', 'ounce', 'hour', 'minute', 'day', 'week',
        'month', 'year', 'dollar', '$', 'percent', '%', 'model', 'version',
        'size', 'color', 'weight', 'battery', 'shipped', 'arrived', 'returned',
        'warranty', 'refund', 'customer service', 'delivery', 'packaging'
    ]
    features['specificity_count'] = sum(1 for w in specific_indicators if w in text_lower)

    features['number_count'] = len(re.findall(r'\d+', text))

    hedging_words = [
        'maybe', 'perhaps', 'somewhat', 'slightly', 'a bit', 'kind of',
        'sort of', 'fairly', 'reasonably', 'adequate', 'decent', 'okay',
        'alright', 'not bad', 'could be better', 'minor', 'small issue',
        'however', 'although', 'but', 'though', 'despite', 'except'
    ]
    features['hedging_count'] = sum(1 for w in hedging_words if w in text_lower)

    return features
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'No data received'}), 400

        review_text = data.get('review', '')

        if not review_text.strip():
            return jsonify({'error': 'Please enter a review text'}), 400

        print(f"Analyzing: \"{review_text[:80]}...\"")

        
        processed = preprocess_text(review_text)

        tfidf_features = tfidf.transform([processed])

        
        num_features = extract_features(review_text)
        num_df = pd.DataFrame([num_features])

        
        combined = hstack([tfidf_features, num_df.values])

        
        prediction = model.predict(combined)[0]
        probability = model.predict_proba(combined)[0]

        fake_prob = round(float(probability[1]) * 100, 1)
        real_prob = round(float(probability[0]) * 100, 1)
        is_fake = bool(prediction == 1)

        print(f"   Result: {' FAKE' if is_fake else ' GENUINE'} ({max(fake_prob, real_prob)}%)")

        
        analysis = {
            'word_count': int(num_features['word_count']),
            'exclamation_marks': int(num_features['exclamation_count']),
            'capital_letter_ratio': round(float(num_features['capital_ratio']) * 100, 1),
            'unique_word_ratio': round(float(num_features['unique_word_ratio']), 2),
            'suspicious_words': int(num_features['fake_word_count']),
            'marketing_words': int(num_features['marketing_word_count']),
            'all_caps_words': int(num_features['all_caps_words']),
            'sentiment_polarity': round(float(num_features['polarity']), 2),
            'sentiment_subjectivity': round(float(num_features['subjectivity']), 2),
            'specificity_score': int(num_features['specificity_count']),
            'hedging_words': int(num_features['hedging_count'])
        }

        result = {
            'prediction': 'FAKE' if is_fake else 'GENUINE',
            'is_fake': is_fake,
            'confidence': float(fake_prob if is_fake else real_prob),
            'fake_probability': float(fake_prob),
            'real_probability': float(real_prob),
            'analysis': analysis
        }

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in /analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running!'})
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Server running at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=5000)