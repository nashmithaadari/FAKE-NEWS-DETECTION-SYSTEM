import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🔍 FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

# Step 1: Load the processed data
print("\n📂 Loading processed data...")
df = pd.read_csv('data/processed_news.csv')
print(f"✅ Loaded {len(df)} articles")

# Step 2: Text Preprocessing Function
print("\n🧹 Cleaning text data...")

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply cleaning to the full_text column
df['cleaned_text'] = df['full_text'].apply(clean_text)
print("✅ Text cleaning complete")

# Step 3: Split data into training and testing sets
print("\n📊 Splitting data...")
X = df['cleaned_text']  # Features (news text)
y = df['label']         # Target (0=Fake, 1=Real)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} articles")
print(f"Testing set: {len(X_test)} articles")

# Step 4: Convert text to numbers using TF-IDF
print("\n🔄 Converting text to numerical features...")
tfidf = TfidfVectorizer(
    max_features=5000,  # Use top 5000 words
    ngram_range=(1, 2),  # Use single words and pairs of words
    stop_words='english'  # Remove common words like 'the', 'and'
)

# Transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"✅ Features shape: {X_train_tfidf.shape}")

# Step 5: Train the model
print("\n🤖 Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0  # Regularization parameter
)

model.fit(X_train_tfidf, y_train)
print("✅ Model training complete!")

# Step 6: Evaluate the model
print("\n📈 Model Performance:")
print("-" * 40)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# Detailed report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['FAKE (0)', 'REAL (1)']))

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print("                 Predicted")
print("                 FAKE  REAL")
cm = confusion_matrix(y_test, y_pred)
print(f"Actual  FAKE:    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"        REAL:    {cm[1,0]:4d}  {cm[1,1]:4d}")

# Step 7: Save the model and vectorizer
print("\n💾 Saving model and vectorizer...")
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("✅ Model saved to 'models/fake_news_model.pkl'")
print("✅ Vectorizer saved to 'models/tfidf_vectorizer.pkl'")

# Step 8: Test with sample predictions
print("\n🔮 Sample Predictions:")
print("-" * 40)

# Get 5 random samples from test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for idx in sample_indices:
    text = X_test.iloc[idx][:200]  # First 200 chars
    true_label = y_test.iloc[idx]
    pred_label = model.predict(tfidf.transform([X_test.iloc[idx]]))[0]
    
    true_label_text = "REAL" if true_label == 1 else "FAKE"
    pred_label_text = "REAL" if pred_label == 1 else "FAKE"
    
    status = "✅" if true_label == pred_label else "❌"
    
    print(f"\n{status} News: {text[:100]}...")
    print(f"   Actual: {true_label_text} | Predicted: {pred_label_text}")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Run this script to train the model")
print("2. We'll create a web interface next")
print("3. Then you can test with your own news articles")