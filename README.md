🧠 TrustLens NLP
AI-Powered Fake Review Detection System

TrustLens NLP is a full-stack Natural Language Processing web application that detects whether a product review is FAKE 🚨 or GENUINE ✅ using advanced machine learning and linguistic feature engineering.

🚀 Live Project Overview

This system:
Uses TF-IDF (1–3 grams)
Extracts 25+ custom linguistic & behavioral features
Applies sentiment analysis (TextBlob)
Uses an Ensemble Voting Classifier
Provides confidence scores
Returns detailed review analytics

🧠 Model Architecture
🔠 Text Features
TF-IDF Vectorizer
1–3 gram range
15,000 max features
Sublinear TF scaling

🧮 Engineered Features
Word count
Character count
Exclamation frequency
Capital letter ratio
Unique word ratio
Word repetition detection
Fake keyword frequency
Marketing phrase detection
Specificity indicators
Hedging words detection
Sentiment polarity
Subjectivity score

🤖 Ensemble Models
Logistic Regression (balanced)
Random Forest (200 trees)
Gradient Boosting
Soft Voting Classifier

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/TrustLens-NLP.git
cd TrustLens-NLP

2️⃣ Install dependencies
pip install flask scikit-learn nltk textblob pandas numpy scipy joblib

3️⃣ Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

<img width="1065" height="5724" alt="127 0 0 1_5000_ (1)" src="https://github.com/user-attachments/assets/e5f0d343-614e-437f-a442-630ab9d49c4a" />
<img width="1065" height="5724" alt="127 0 0 1_5000_ (2)" src="https://github.com/user-attachments/assets/133bfd14-6620-4f4a-a224-d6025b33786b" />

🔮 Future Improvements
Transformer-based model (BERT)
Real-world dataset integration
Docker deployment
AWS/GCP cloud hosting
Admin analytics dashboard







